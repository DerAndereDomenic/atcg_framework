#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Integrator/PathtracingData.cuh>

#include <Core/TraceParameters.h>
#include <Core/SurfaceInteraction.h>
#include <Core/Payload.h>
#include <Math/Random.h>

#include <Spectrum/SampledSpectrum.h>

extern "C"
{
    __constant__ atcg::PathtracingParams params;
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

    atcg::SurfaceInteraction last_si;
    float last_bsdf_pdf = 1.0f;

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

        if(si.valid && n == 0)
        {
            entity_id = si.entity_id;
        }

        if(si.valid)
        {
            // Check for light source
            if(si.emitter)
            {
                bool mis_valid             = last_si.valid;
                float emitter_sampling_pdf = mis_valid ? si.emitter->evalLightSamplingPdf(last_si, si) : 0.0f;
                float mis_weight           = last_bsdf_pdf / (last_bsdf_pdf + emitter_sampling_pdf);
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
                                bsdf_result.bsdf_value *
                                glm::abs(glm::dot(si.normal, emitter_sampling.direction_to_light));
                } while(false);

                auto result = si.bsdf->sampleBSDF(si, wavelengths, rng);

                if(result.sample_probability > 0.0f)
                {
                    camera_ray.ray.origin    = si.position;
                    camera_ray.ray.direction = result.out_dir;
                    camera_ray.importance *= result.bsdf_weight;
                    next_ray_valid = true;

                    last_si       = si;
                    last_bsdf_pdf = result.sample_probability;

                    if((int)(result.flags & atcg::BSDFComponentType::AnyDelta) != 0)
                    {
                        last_si.valid = false;
                    }
                }
            }
        }
        else
        {
            if(params.environment_emitter)
            {
                bool mis_valid              = last_si.valid;
                float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);
                float emitter_sampling_pdf =
                    mis_valid ? params.environment_emitter->evalLightSamplingPdf(last_si, si) * emitter_selection_pdf
                              : 0.0f;
                float mis_weight = last_bsdf_pdf / (last_bsdf_pdf + emitter_sampling_pdf);
                radiance += mis_weight * camera_ray.importance * params.environment_emitter->evalLight(si, wavelengths);
            }
        }
    }

    params.sensor->addSample(glm::ivec3(launch_idx.x, launch_idx.y, params.frame_counter), radiance, wavelengths);

    // if(params.frame_counter > 0)
    // {
    //     // Mix with previous subframes if present!
    //     const float a                        = 1.0f / static_cast<float>(params.frame_counter + 1);
    //     const glm::vec3 prev_output_radiance = params.accumulation_buffer[pixel_index];
    //     radiance                             = glm::lerp(prev_output_radiance, radiance, a);
    // }

    // params.accumulation_buffer[pixel_index] = radiance;

    // atcg::SampledSpectrum white(1.0f);
    // auto white_lrgb = atcg::Color::XYZ_to_lRGB(white.toXYZ(wavelengths));

    // glm::vec3 xyz  = radiance.toXYZ(wavelengths);
    // glm::vec3 lrgb = atcg::Color::XYZ_to_lRGB(xyz) / white_lrgb;

    // glm::vec3 mapped = glm::vec3(1.0f) - glm::exp(-lrgb);

    // glm::vec3 srgb = atcg::Color::lRGB_to_sRGB(mapped);

    // srgb.x = glm::min(glm::max(srgb.x, 0.0f), 1.0f);
    // srgb.y = glm::min(glm::max(srgb.y, 0.0f), 1.0f);
    // srgb.z = glm::min(glm::max(srgb.z, 0.0f), 1.0f);

    // glm::u8vec3 quantized = atcg::Color::quantize(srgb);

    // params.output_image[pixel_index] = glm::u8vec4(quantized, 255);

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

    si->valid              = false;
    si->incoming_distance  = std::numeric_limits<float>::infinity();
    si->incoming_direction = ray_dir;
}

extern "C" __global__ void __miss__occlusion()
{
    setOcclusionPayload(false);
}