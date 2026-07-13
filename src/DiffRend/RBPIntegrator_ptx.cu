#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include "RBPData.cuh"

#include <Core/TraceParameters.h>
#include <Core/SurfaceInteraction.h>
#include <Core/Payload.h>
#include <Math/Random.h>
#include <Utils/HostDevice.h>
#include <DataStructure/Ray.h>

#define NUM_BOUNCES 8

extern "C"
{
    __constant__ atcg::RBPParams params;
}

struct RayContext
{
    bool valid;
    glm::vec3 origin;
    glm::vec3 direction;
    glm::vec3 throughput;
    glm::vec3 radiance;
};

ATCG_INLINE ATCG_DEVICE glm::vec3 Li(const atcg::Ray& ray_,
                                     const atcg::SurfaceInteraction& last_si_,
                                     const atcg::SampledWavelengths& wavelengths,
                                     int n_max,
                                     atcg::PCG32& rng)
{
    RayContext ray;

    ray.direction  = ray_.direction;
    ray.origin     = ray_.origin;
    ray.radiance   = glm::vec3(0.0f);
    ray.throughput = glm::vec3(1);
    ray.valid      = true;

    glm::vec3 next_origin;
    glm::vec3 next_dir;

    atcg::SurfaceInteraction last_si = last_si_;

    for(int n = 0; n < n_max; ++n)
    {
        if(!ray.valid) return ray.radiance;
        ray.valid = false;

        // float rr_prob = glm::max(glm::max(ray.throughput.r, ray.throughput.g), ray.throughput.b);
        // if(rng.nextFloat() < rr_prob)
        // {
        //     ray.throughput /= rr_prob;
        // }
        // else
        // {
        //     return ray.radiance;
        // }

        atcg::SurfaceInteraction si;
        atcg::traceWithDataPointer<atcg::SurfaceInteraction>(params.handle,
                                                             ray.origin,
                                                             ray.direction,
                                                             0.001f,
                                                             1e16f,
                                                             &si,
                                                             params.surface_trace_params);
        if(si.isValid())
        {
            // Check for light source
            if(si.emitter)
            {
                bool mis_valid              = last_si.isValid();
                float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);
                float emitter_sampling_pdf =
                    atcg::select(mis_valid,
                                 si.emitter->evalLightSamplingPdf(last_si, si) * emitter_selection_pdf,
                                 0.0f);
                float mis_weight = last_si.pdf / (last_si.pdf + emitter_sampling_pdf);

                ray.radiance += mis_weight * ray.throughput * si.emitter->evalLight(si, wavelengths);
            }

            // PBR Sampling
            if(si.bsdf)
            {
                // Next-event estimation
                do
                {
                    if(params.num_emitters == 0) break;

                    uint32_t emitter_index = rng.nextUint32() % params.num_emitters;

                    float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);

                    const atcg::EmitterVPtrTable* emitter = params.emitters[emitter_index];

                    if(si.emitter == emitter) break;

                    atcg::EmitterSamplingResult emitter_sampling = emitter->sampleLight(si, wavelengths, rng);

                    if(emitter_sampling.sampling_pdf == 0) break;

                    emitter_sampling.sampling_pdf *= emitter_selection_pdf;
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

                    float bsdf_pdf = atcg::select((int)(emitter->flags & atcg::EmitterFlags::InfinitesimalSize) != 0 ||
                                                      (int)(bsdf_result.flags & atcg::BSDFComponentType::AnyDelta) != 0,
                                                  0.0f,
                                                  bsdf_result.sample_probability);
                    float mis_weight = emitter_sampling.sampling_pdf / (emitter_sampling.sampling_pdf + bsdf_pdf);

                    glm::vec3 radiance_nee = mis_weight * ray.throughput *
                                             emitter_sampling.radiance_weight_at_receiver * bsdf_result.bsdf_value;

                    ray.radiance += radiance_nee;

                } while(false);

                auto result = si.bsdf->sampleBSDF(si, wavelengths, rng);

                if(result.sample_probability > 0.0f)
                {
                    next_origin = si.position;
                    next_dir    = result.out_dir;
                    ray.throughput *= glm::vec3(result.bsdf_weight);
                    ray.valid = true;

                    last_si     = si;
                    last_si.pdf = result.sample_probability;

                    if((int)(result.flags & atcg::BSDFComponentType::AnyDelta) != 0)
                    {
                        last_si.setInvalid();
                    }
                }
            }
        }
        else
        {
            if(params.environment_emitter)
            {
                bool mis_valid              = last_si.isValid();
                float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);
                float emitter_sampling_pdf =
                    atcg::select(mis_valid,
                                 params.environment_emitter->evalLightSamplingPdf(last_si, si) * emitter_selection_pdf,
                                 0.0f);
                float mis_weight = last_si.pdf / (last_si.pdf + emitter_sampling_pdf);

                ray.radiance += mis_weight * ray.throughput * params.environment_emitter->evalLight(si, wavelengths);
            }
        }

        ray.origin    = next_origin;
        ray.direction = next_dir;
    }

    return ray.radiance;
}

ATCG_INLINE ATCG_DEVICE void dLi(const glm::vec3& grad_out,
                                 const atcg::Ray& ray_,
                                 const atcg::SampledWavelengths& wavelengths,
                                 int n_max,
                                 atcg::PCG32& rng)
{
    RayContext ray;

    ray.direction  = ray_.direction;
    ray.origin     = ray_.origin;
    ray.throughput = grad_out;
    ray.valid      = true;

    glm::vec3 next_origin;
    glm::vec3 next_dir;

    atcg::SurfaceInteraction last_si;
    last_si.pdf = 1.0f;

    for(int n = 0; n < n_max; ++n)
    {
        if(!ray.valid) return;
        ray.valid = false;

        // float rr_prob = glm::max(glm::max(ray.throughput.r, ray.throughput.g), ray.throughput.b);
        // if(rng.nextFloat() < rr_prob)
        // {
        //     ray.throughput /= rr_prob;
        // }
        // else
        // {
        //     return;
        // }

        atcg::SurfaceInteraction si;
        atcg::traceWithDataPointer<atcg::SurfaceInteraction>(params.handle,
                                                             ray.origin,
                                                             ray.direction,
                                                             0.001f,
                                                             1e16f,
                                                             &si,
                                                             params.surface_trace_params);
        if(si.isValid())
        {
            // PBR Sampling
            if(si.bsdf)
            {
                // Next-event estimation
                do
                {
                    if(params.num_emitters == 0) break;

                    uint32_t emitter_index = rng.nextUint32() % params.num_emitters;

                    float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);

                    const atcg::EmitterVPtrTable* emitter = params.emitters[emitter_index];

                    if(si.emitter == emitter) break;

                    atcg::EmitterSamplingResult emitter_sampling = emitter->sampleLight(si, wavelengths, rng);

                    if(emitter_sampling.sampling_pdf == 0) break;

                    emitter_sampling.sampling_pdf *= emitter_selection_pdf;
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

                    float bsdf_pdf = atcg::select((int)(emitter->flags & atcg::EmitterFlags::InfinitesimalSize) != 0 ||
                                                      (int)(bsdf_result.flags & atcg::BSDFComponentType::AnyDelta) != 0,
                                                  0.0f,
                                                  bsdf_result.sample_probability);
                    float mis_weight = emitter_sampling.sampling_pdf / (emitter_sampling.sampling_pdf + bsdf_pdf);

                    glm::vec3 radiance_nee = mis_weight * ray.throughput *
                                             emitter_sampling.radiance_weight_at_receiver * bsdf_result.bsdf_value;

                    glm::vec3 g = mis_weight * ray.throughput * emitter_sampling.radiance_weight_at_receiver;
                    si.bsdf->evalBSDFBackward(si, emitter_sampling.direction_to_light, g);

                } while(false);

                auto result = si.bsdf->sampleBSDF(si, wavelengths, rng);

                if(result.sample_probability > 0.0f)
                {
                    glm::vec3 Li_ = Li(atcg::Ray(si.position, result.out_dir),
                                       last_si,
                                       wavelengths,
                                       NUM_BOUNCES - n - 1,
                                       rng);    // O(n^2)

                    glm::vec3 g = ray.throughput / result.sample_probability * Li_;

                    si.bsdf->evalBSDFBackward(si, result.out_dir, g);

                    next_origin = si.position;
                    next_dir    = result.out_dir;
                    ray.throughput *= glm::vec3(result.bsdf_weight);
                    ray.valid = true;

                    last_si     = si;
                    last_si.pdf = result.sample_probability;

                    if((int)(result.flags & atcg::BSDFComponentType::AnyDelta) != 0)
                    {
                        last_si.setInvalid();
                    }
                }
            }
        }


        ray.origin    = next_origin;
        ray.direction = next_dir;
    }
}

extern "C" __global__ void __raygen__forward()
{
    uint3 launch_idx = optixGetLaunchIndex();

    uint32_t pixel_index = launch_idx.x + params.image_width * launch_idx.y;
    uint64_t seed        = atcg::sampleTEA64(pixel_index, params.rng_index);
    atcg::PCG32 rng(seed);

    atcg::SampledWavelengths wavelengths = atcg::SampledWavelengths::sampleSpectrum(rng.next1d(), 380.0f, 780.0f);


    atcg::CameraRay camera_ray = params.sensor->generateRay(glm::ivec2(launch_idx.x, launch_idx.y), rng);

    atcg::Ray ray = camera_ray.ray;

    atcg::SurfaceInteraction last_si;
    last_si.pdf = 1.0f;

    glm::vec3 radiance = Li(ray, last_si, wavelengths, NUM_BOUNCES, rng);

    params.current_sample[pixel_index] = radiance;
}

extern "C" __global__ void __raygen__backward()
{
    uint3 launch_idx = optixGetLaunchIndex();

    uint32_t pixel_index = launch_idx.x + params.image_width * launch_idx.y;
    uint64_t seed        = atcg::sampleTEA64(pixel_index, params.rng_index);
    atcg::PCG32 rng(seed);

    atcg::SampledWavelengths wavelengths = atcg::SampledWavelengths::sampleSpectrum(rng.next1d(), 380.0f, 780.0f);

    atcg::CameraRay camera_ray = params.sensor->generateRay(glm::ivec2(launch_idx.x, launch_idx.y), rng);
    atcg::Ray ray              = camera_ray.ray;

    dLi(params.adjoint_y[pixel_index], ray, wavelengths, NUM_BOUNCES, rng);
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