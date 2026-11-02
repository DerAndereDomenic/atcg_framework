#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include "EmitterIntegratorData.cuh"

#include <Core/TraceParameters.h>
#include <Core/SurfaceInteraction.h>
#include <Core/Payload.h>
#include <Math/Random.h>

#include <Spectrum/SampledSpectrum.h>
#include <Integrator/MIS.h>

extern "C"
{
    __constant__ atcg::EmitterPathtracingParams params;
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

    atcg::SampledSpectrum radiance(0);
    atcg::SampledWavelengths wavelengths = atcg::SampledWavelengths::sampleSpectrum(rng.nextFloat(), 380.0f, 780.0f);
    int32_t entity_id                    = -1;

    bool next_ray_valid = true;


    for(int n = 0; n < 512; ++n)
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

        if(si.isValid())
        {
            // Check for light source
            if(n == 0 && si.emitter)
            {
                radiance += camera_ray.importance * si.emitter->evalLight(si, wavelengths);
            }

            // PBR Sampling
            if(!si.bsdf)
            {
                next_ray_valid = false;
                continue;
            }

            // Next-event estimation
            do
            {
                if(params.num_emitters == 0) break;

                uint32_t emitter_index = rng.nextUint32() % params.num_emitters;

                float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);

                const atcg::EmitterVPtrTable* emitter = params.emitters[emitter_index];

                if(si.emitter == emitter) break;

                atcg::EmitterSamplingResult emitter_sampling = emitter->sampleLight(si, wavelengths, rng);

                if(emitter_sampling.pdf_dw == 0) break;

                emitter_sampling.pdf_dw *= emitter_selection_pdf;
                emitter_sampling.radiance_weight_at_receiver /= emitter_selection_pdf;

                bool occluded = atcg::traceOcclusion(params.handle,
                                                     si.position,
                                                     emitter_sampling.direction_to_light,
                                                     1e-4f,
                                                     emitter_sampling.distance_to_light - 1e-4f);

                if(occluded)
                {
                    break;
                }

                atcg::BSDFEvalResult bsdf_result =
                    si.bsdf->evalBSDF(si, emitter_sampling.direction_to_light, wavelengths);

                radiance +=
                    camera_ray.importance * emitter_sampling.radiance_weight_at_receiver * bsdf_result.bsdf_value;
            } while(false);

            auto result = si.bsdf->sampleBSDF(si, wavelengths, rng);
            if(result.pdf_dw > 0.0f)
            {
                camera_ray.importance *= result.bsdf_weight;
                camera_ray.ray.origin    = si.position;
                camera_ray.ray.direction = result.out_dir;

                next_ray_valid = true;
            }
        }
        else
        {
            if(n == 0 && params.environment_emitter)
            {
                radiance += camera_ray.importance * params.environment_emitter->evalLight(si, wavelengths);
            }
        }
    }

    params.sensor->addSample(glm::ivec3(launch_idx.x, launch_idx.y, params.frame_counter), radiance, wavelengths);

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