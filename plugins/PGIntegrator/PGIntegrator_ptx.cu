#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include "PGIntegratorData.cuh"

#include <Core/TraceParameters.h>
#include <Core/SurfaceInteraction.h>
#include <Core/Payload.h>
#include <Math/Random.h>

#include <Spectrum/SampledSpectrum.h>
#include <Integrator/MIS.h>

extern "C"
{
    __constant__ atcg::PGIntegratorParams params;
}

struct Sample
{
    int vmf_index;
    glm::vec3 direction;
};

ATCG_INLINE ATCG_DEVICE glm::vec3 clampVec(const glm::vec3& value, float min, float max)
{
    return glm::vec3(glm::clamp(value.x, min, max), glm::clamp(value.y, min, max), glm::clamp(value.z, min, max));
}

ATCG_INLINE ATCG_DEVICE glm::vec3 clampVec(const glm::vec3& value, const glm::vec3& min, const glm::vec3& max)
{
    return glm::vec3(glm::clamp(value.x, min.x, max.x),
                     glm::clamp(value.y, min.y, max.y),
                     glm::clamp(value.z, min.z, max.z));
}

ATCG_INLINE ATCG_DEVICE glm::vec3 mapToBoundingBox(const glm::vec3& pos)
{
    glm::vec3 bbox_min = params.scene_aabb->min;
    glm::vec3 bbox_max = params.scene_aabb->max;

    return clampVec((pos - (bbox_min)) / (bbox_max - bbox_min), 0.0f, 1.0f);
}

ATCG_INLINE ATCG_DEVICE glm::ivec3 mapToGrid(const glm::vec3& pos, const glm::ivec3& grid_size)
{
    glm::vec3 mapped_pos = mapToBoundingBox(pos);
    return glm::ivec3(
        clampVec(glm::floor(mapped_pos * glm::vec3(grid_size)), glm::vec3(0), glm::vec3(grid_size) - 1.0f));
}

ATCG_INLINE ATCG_DEVICE int mapToGridIndex(const glm::ivec3& grid_index, const glm::ivec3& grid_size)
{
    return grid_index.x + grid_size.x * (grid_index.y + grid_size.y * grid_index.z);
}

extern "C" __global__ void __raygen__rg()
{
    uint3 launch_idx = optixGetLaunchIndex();

    uint32_t pixel_index = launch_idx.x + params.image_width * launch_idx.y;
    uint64_t seed        = atcg::sampleTEA64(pixel_index, params.frame_counter);
    atcg::PCG32 rng(seed);

    atcg::CameraRay camera_ray = params.sensor->generateRay(glm::ivec2(launch_idx.x, launch_idx.y), rng);

    if(!camera_ray.valid)
    {
        return;
    }

    const glm::ivec3 grid_size(64, 64, 64);

    atcg::SampledSpectrum radiance(0);
    atcg::SampledWavelengths wavelengths = atcg::SampledWavelengths::sampleSpectrum(rng.nextFloat(), 380.0f, 780.0f);
    int32_t entity_id                    = -1;

    bool next_ray_valid = true;

    atcg::SurfaceInteraction last_si;
    last_si.pdf = 1.0f;

    Sample samples[8];
    memset(samples, 0, sizeof(Sample) * 8);

    for(int n = 0; n < 8; ++n)
    {
        if(!next_ray_valid) break;
        next_ray_valid = false;

        atcg::SurfaceInteraction si;
        atcg::traceWithDataPointer<atcg::SurfaceInteraction>(params.handle,
                                                             camera_ray.ray.origin,
                                                             camera_ray.ray.direction,
                                                             0.001f,
                                                             1e16f,
                                                             &si,
                                                             params.surface_trace_params);

        if(si.isValid() && n == 0)
        {
            entity_id = si.entity_id;
        }

        if(!si.isValid())
        {
            if(params.environment_emitter)
            {
                bool mis_valid              = last_si.isValid();
                float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);
                float emitter_sampling_pdf =
                    mis_valid ? params.environment_emitter->evalLightSamplingPdf(last_si, si) * emitter_selection_pdf
                              : 0.0f;
                float mis_weight = 1.0f;    // last_si.pdf / (last_si.pdf + emitter_sampling_pdf);
                radiance += mis_weight * camera_ray.importance * params.environment_emitter->evalLight(si, wavelengths);
            }
            break;
        }

        glm::vec3 normalized_pos = mapToBoundingBox(si.position);
        glm::ivec3 grid_index    = mapToGrid(normalized_pos, grid_size);
        int vmf_index            = mapToGridIndex(grid_index, grid_size);

        atcg::vMFMixture<5>& vmf_mixture = params.vmf_mixture[vmf_index];

        // Check for light source
        if(si.emitter)
        {
            bool mis_valid              = last_si.isValid();
            float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);
            float emitter_sampling_pdf =
                mis_valid ? si.emitter->evalLightSamplingPdf(last_si, si) * emitter_selection_pdf : 0.0f;
            float mis_weight = 1.0f;    // atcg::BalanceHeuristic::apply(last_si.pdf, emitter_sampling_pdf);
            radiance += mis_weight * camera_ray.importance * si.emitter->evalLight(si, wavelengths);

            if(!params.use_guiding)
            {
                for(int i = 0; i < n; ++i)
                {
                    atcg::vMFMixture<5>* vmf = &params.vmf_mixture[samples[i].vmf_index];
                    vmf->accumulate(samples[i].direction);
                }
            }
        }

        // PBR Sampling
        if(si.bsdf)
        {
            // Next-event estimation
            do
            {
                break;
                if(params.num_emitters == 0) break;

                uint32_t emitter_index = rng.nextUint32() % params.num_emitters;

                float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);

                const atcg::EmitterVPtrTable* emitter = params.emitters[emitter_index];

                if(si.emitter == emitter) break;

                atcg::EmitterSamplingResult emitter_sampling = emitter->sampleLight(si, wavelengths, rng);

                if(emitter_sampling.pdf_dw == 0) break;

                emitter_sampling.pdf_dw *= emitter_selection_pdf;
                emitter_sampling.radiance_weight_at_receiver /= emitter_selection_pdf;

                bool occluded = traceOcclusion(params.handle,
                                               si.position,
                                               emitter_sampling.direction_to_light,
                                               1e-3f,
                                               emitter_sampling.distance_to_light - 1e-3f,
                                               params.occlusion_trace_params);

                if(occluded)
                {
                    break;
                }

                atcg::BSDFEvalResult bsdf_result =
                    si.bsdf->evalBSDF(si, emitter_sampling.direction_to_light, wavelengths);

                float bsdf_pdf   = (int)(emitter->flags & atcg::EmitterFlags::InfinitesimalSize) != 0 ||
                                           (int)(bsdf_result.flags & atcg::BSDFComponentType::AnyDelta) != 0
                                       ? 0.0f
                                       : bsdf_result.pdf_dw;
                float mis_weight = atcg::BalanceHeuristic::apply(emitter_sampling.pdf_dw, bsdf_pdf);

                radiance += mis_weight * camera_ray.importance * emitter_sampling.radiance_weight_at_receiver *
                            bsdf_result.bsdf_value;
            } while(false);

            atcg::BSDFSamplingResult result;

            if(params.use_guiding)
            {
                glm::vec3 out_dir = vmf_mixture.sample(rng);
                result.pdf_dw     = 0.0f;

                if(glm::dot(out_dir, si.normal) > 0.0f)
                {
                    float pdf = vmf_mixture.eval(out_dir);

                    auto bsdf_val = si.bsdf->evalBSDF(si, out_dir, wavelengths);

                    result.out_dir     = out_dir;
                    result.bsdf_weight = bsdf_val.bsdf_value / pdf;
                    result.pdf_dw      = pdf;
                }
            }
            else
            {
                result = si.bsdf->sampleBSDF(si, wavelengths, rng);
            }

            if(result.pdf_dw > 0.0f)
            {
                camera_ray.ray.origin    = si.position;
                camera_ray.ray.direction = result.out_dir;
                camera_ray.importance *= result.bsdf_weight;
                next_ray_valid = true;

                glm::vec3 normalized_pos = mapToBoundingBox(si.position);
                glm::ivec3 grid_index    = mapToGrid(normalized_pos, grid_size);
                int vmf_index            = mapToGridIndex(grid_index, grid_size);

                samples[n].vmf_index = vmf_index;
                samples[n].direction = result.out_dir;

                last_si     = si;
                last_si.pdf = result.pdf_dw;

                if((int)(result.flags & atcg::BSDFComponentType::AnyDelta) != 0)
                {
                    last_si.setInvalid();
                }
            }
        }
    }

    // TODO No accumulation for now
    if(params.accumulate)
    {
        params.sensor->addSample(glm::ivec3(launch_idx.x, launch_idx.y, params.frame_counter), radiance, wavelengths);
    }
    else
    {
        params.sensor->addSample(glm::ivec3(launch_idx.x, launch_idx.y, 0), radiance, wavelengths);
    }

    if(params.entity_ids)
    {
        params.entity_ids[pixel_index] = entity_id;
    }
}

extern "C" __global__ void __miss__ms()
{
    atcg::SurfaceInteraction* si = getPayloadDataPointer<atcg::SurfaceInteraction>();
    float3 optix_world_dir       = optixGetWorldRayDirection();
    glm::vec3 ray_dir            = glm::make_vec3((float*)&optix_world_dir);

    si->setInvalid();
    si->incoming_direction = ray_dir;
}

extern "C" __global__ void __miss__occlusion()
{
    setOcclusionPayload(false);
}