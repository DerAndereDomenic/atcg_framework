#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Integrator/VolPathtracingData.cuh>

#include <Core/TraceParameters.h>
#include <Core/SurfaceInteraction.h>
#include <Core/Payload.h>
#include <Math/Random.h>

extern "C"
{
    __constant__ atcg::VolPathtracingParams params;
}

extern "C" __global__ void __raygen__rg()
{
    uint3 launch_idx = optixGetLaunchIndex();

    uint32_t pixel_index = launch_idx.x + params.image_width * launch_idx.y;
    uint64_t seed        = atcg::sampleTEA64(pixel_index, params.frame_counter);
    atcg::PCG32 rng(seed);

    glm::vec2 jitter = rng.next2d();
    float u          = (((float)launch_idx.x + jitter.x) / (float)params.image_width - 0.5f) * 2.0f;
    float v          = (((float)launch_idx.y + jitter.y) / (float)params.image_height - 0.5f) * 2.0f;

    atcg::CameraRay camera_ray = params.sensor->generateRay(glm::vec2(u, v));

    atcg::SampledSpectrum radiance(0);
    atcg::SampledWavelengths wavelengths = atcg::SampledWavelengths::sampleSpectrum(rng.nextFloat(), 380.0f, 780.0f);
    int32_t entity_id                    = -1;

    bool next_ray_valid = true;

    atcg::AnyInteraction last_ai;
    last_ai->pdf = 1.0f;

    for(int n = 0; n < 512; ++n)
    {
        if(!next_ray_valid) break;
        next_ray_valid = false;

        float rr_prob = camera_ray.importance.maxComponent();
        if(rng.nextFloat() < rr_prob)
        {
            camera_ray.importance /= rr_prob;
        }
        else
        {
            next_ray_valid = false;
            break;
        }

        atcg::SurfaceInteraction si;
        atcg::traceWithDataPointer<atcg::SurfaceInteraction>(params.handle,
                                                             camera_ray.ray.origin,
                                                             camera_ray.ray.direction,
                                                             0.00001f,
                                                             1e16f,
                                                             &si,
                                                             params.surface_trace_params);
        if(si.isValid() &&
           camera_ray.ray.current_medium)    // For now, we only allow media inside objects. So if si not valid, reject
        {
            float max_distance = glm::length(si.position - camera_ray.ray.origin);
            atcg::MediumSamplingResult result =
                camera_ray.ray.current_medium->sampleMediumEvent(camera_ray.ray.origin,
                                                                 camera_ray.ray.direction,
                                                                 max_distance,
                                                                 wavelengths,
                                                                 rng);

            radiance += camera_ray.importance * result.radiance_weight;
            camera_ray.importance *= result.transmittance_weight;

            // Check if a medium event was sampled. Otherwise, skip to surface rendering
            if(result.interaction.isValid())
            {
                atcg::MediumInteraction mi = result.interaction;

                // NEE
                do
                {
                    if(!si.bsdf || (int)(si.bsdf->flags & atcg::BSDFComponentType::NullTransmission) == 0)
                    {
                        break;
                    }
                    if(params.num_emitters == 0) break;

                    uint32_t emitter_index = rng.nextUint32() % params.num_emitters;

                    float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);

                    const atcg::EmitterVPtrTable* emitter = params.emitters[emitter_index];

                    atcg::EmitterSamplingResult emitter_sampling = emitter->sampleLight(mi, wavelengths, rng);

                    if(emitter_sampling.sampling_pdf == 0)
                    {
                        break;
                    }
                    emitter_sampling.sampling_pdf *= emitter_selection_pdf;

                    atcg::SurfaceInteraction si_dummy;
                    atcg::traceWithDataPointer<atcg::SurfaceInteraction>(params.handle,
                                                                         mi.position,
                                                                         emitter_sampling.direction_to_light,
                                                                         0.0f,
                                                                         1e16f,
                                                                         &si_dummy,
                                                                         params.surface_trace_params);

                    if(!si_dummy.isValid())
                    {
                        // Should not happen because we are inside the geometry
                        break;
                    }

                    bool occluded =
                        traceOcclusion(params.handle,
                                       si_dummy.position,
                                       emitter_sampling.direction_to_light,
                                       1e-3f,
                                       emitter_sampling.distance_to_light - si_dummy.incoming_distance - 1e-3f,
                                       params.occlusion_trace_params);

                    if(occluded)
                    {
                        break;
                    }

                    float transmittance_to_light =
                        camera_ray.ray.current_medium->evalTransmittance(mi.position,
                                                                         emitter_sampling.direction_to_light,
                                                                         si_dummy.incoming_distance,
                                                                         rng);

                    auto phase_result = camera_ray.ray.current_medium->phase_function->evalPhaseFunction(
                        mi,
                        emitter_sampling.direction_to_light);
                    float phase_pdf    = phase_result.sampling_pdf;
                    float sampling_pdf = (int)(emitter->flags & atcg::EmitterFlags::InfinitesimalSize) != 0
                                             ? 0.0f
                                             : phase_pdf;    // * transmittance_to_light;

                    float mis_weight = emitter_sampling.sampling_pdf / (emitter_sampling.sampling_pdf + sampling_pdf);

                    radiance += mis_weight * camera_ray.importance * transmittance_to_light *
                                phase_result.phase_function_value * emitter_sampling.radiance_weight_at_receiver;
                } while(false);

                const atcg::PhaseFunctionVPtrTable* phase_function = camera_ray.ray.current_medium->phase_function;
                atcg::PhaseFunctionSamplingResult phase_result     = phase_function->samplePhaseFunction(mi, rng);
                if(phase_result.sampling_pdf == 0)
                {
                    next_ray_valid = false;
                    last_ai->setInvalid();
                    break;
                }

                camera_ray.ray.origin    = mi.position;
                camera_ray.ray.direction = glm::normalize(phase_result.outgoing_ray_dir);
                camera_ray.importance *= phase_result.phase_function_weight;
                next_ray_valid = true;

                last_ai      = mi;
                last_ai->pdf = phase_result.sampling_pdf;

                continue;
            }
        }

        if(!si.isValid())
        {
            if(params.environment_emitter)
            {
                bool mis_valid              = last_ai->isValid();
                float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);
                float emitter_sampling_pdf =
                    mis_valid ? params.environment_emitter->evalLightSamplingPdf(last_ai, si) * emitter_selection_pdf
                              : 0.0f;
                float mis_weight = last_ai->pdf / (last_ai->pdf + emitter_sampling_pdf);
                radiance += mis_weight * camera_ray.importance * params.environment_emitter->evalLight(si, wavelengths);
            }
            next_ray_valid = false;
            break;
        }

        if(n == 0)
        {
            entity_id = si.entity_id;
        }

        // Check for light source
        if(si.emitter)
        {
            bool mis_valid              = last_ai->isValid();
            float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);
            float emitter_sampling_pdf  = mis_valid ? si.emitter->evalLightSamplingPdf(last_ai, si) : 0.0f;
            float mis_weight            = last_ai->pdf / (last_ai->pdf + emitter_sampling_pdf * emitter_selection_pdf);
            radiance += mis_weight * camera_ray.importance * si.emitter->evalLight(si, wavelengths);
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

                float bsdf_pdf   = (int)(emitter->flags & atcg::EmitterFlags::InfinitesimalSize) != 0
                                       ? 0.0f
                                       : bsdf_result.sample_probability;
                float mis_weight = emitter_sampling.sampling_pdf / (emitter_sampling.sampling_pdf + bsdf_pdf);

                radiance += mis_weight * camera_ray.importance * emitter_sampling.radiance_weight_at_receiver *
                            bsdf_result.bsdf_value;
            } while(false);

            auto result = si.bsdf->sampleBSDF(si, wavelengths, rng);

            if(result.sample_probability > 0.0f)
            {
                camera_ray.ray.origin    = si.position;
                camera_ray.ray.direction = result.out_dir;
                camera_ray.importance *= result.bsdf_weight;
                next_ray_valid = true;

                if((int)(result.flags & atcg::BSDFComponentType::NullTransmission) == 0)
                {
                    // If the sampled component is a null transmission, we don't want to count it because for NEE we
                    // need the last non-null-transportation interaction. This is a bit hacky but it works for now.
                    last_ai      = si;
                    last_ai->pdf = result.sample_probability;

                    if((int)(result.flags & atcg::BSDFComponentType::AnyDelta) != 0)
                    {
                        last_ai->setInvalid();    // Invalidate last_ai to prevent NEE for delta interactions
                    }
                }

                // Check if we are entering the geometry or leaving the geometry and assign si.inside_medium or
                // si.outside_medium, respectively.
                float cos_theta_curr_ray = glm::dot(si.normal, si.incoming_direction);
                float cos_theta_next_ray = glm::dot(si.normal, camera_ray.ray.direction);
                // Only change the medium if we have a transmission...
                if(cos_theta_curr_ray * cos_theta_next_ray > 0)
                {
                    camera_ray.ray.current_medium = cos_theta_next_ray < 0 ? si.inside_medium : si.outside_medium;
                }
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

extern "C" __global__ void __miss__occlusion()
{
    setOcclusionPayload(false);
}