#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Renderer/Params.cuh>

#include <Math/Random.h>
#include <Math/SurfaceInteraction.h>
#include <Renderer/Payload.h>

extern "C"
{
    __constant__ Params params;
}

extern "C" __global__ void __raygen__rg()
{
    uint3 launch_idx = optixGetLaunchIndex();

    uint32_t pixel_index = launch_idx.x + params.image_width * launch_idx.y;
    uint64_t seed        = atcg::sampleTEA64(pixel_index, params.frame_counter);
    atcg::PCG32 rng(seed);

    glm::vec2 jitter = rng.next2d();
    float u          = (((float)launch_idx.x + jitter.x) / (float)params.image_width - 0.5f) * 2.0f;
    float v = (((float)(params.image_height - launch_idx.y) + jitter.y) / (float)params.image_height - 0.5f) * 2.0f;

    glm::vec3 cam_eye = glm::make_vec3(params.cam_eye);
    glm::vec3 U       = glm::make_vec3(params.U) * (float)params.image_width / (float)params.image_height;
    glm::vec3 V       = glm::make_vec3(params.V);
    glm::vec3 W       = glm::make_vec3(params.W);

    glm::vec3 ray_dir    = glm::normalize(u * U + v * V + W);
    glm::vec3 ray_origin = cam_eye;
    glm::vec3 radiance(0);
    glm::vec3 throughput(1);

    glm::vec3 next_origin;
    glm::vec3 next_dir;

    bool next_ray_valid = true;

    for(int n = 0; n < 50; ++n)
    {
        if(!next_ray_valid) break;
        next_ray_valid = false;

        atcg::SurfaceInteraction si;
        traceWithDataPointer<atcg::SurfaceInteraction>(params.handle, ray_origin, ray_dir, 0.001f, 1e16f, &si);

        if(si.valid)
        {
            // Check for light source
            if(si.emitter)
            {
                radiance += throughput * si.emitter->evalLight(si);
            }

            // PBR Sampling
            if(si.bsdf)
            {
                // Next-event estimation
                for(uint32_t i = 0; i < params.num_emitters; ++i)
                {
                    const atcg::EmitterVPtrTable* emitter = params.emitters[i];

                    atcg::EmitterSamplingResult emitter_sampling = emitter->sampleLight(si, rng);

                    if(emitter_sampling.sampling_pdf == 0) continue;

                    atcg::SurfaceInteraction emitter_si;
                    traceWithDataPointer<atcg::SurfaceInteraction>(params.handle,
                                                                   si.position,
                                                                   emitter_sampling.direction_to_light,
                                                                   1e-3f,
                                                                   emitter_sampling.distance_to_light - 1e-3f,
                                                                   &emitter_si);

                    if(!si.valid)
                    {
                        atcg::BSDFEvalResult bsdf_result = si.bsdf->evalBSDF(si, emitter_sampling.direction_to_light);
                        radiance += throughput * emitter_sampling.radiance_weight_at_receiver * bsdf_result.bsdf_value *
                                    glm::max(0.0f, glm::dot(si.normal, emitter_sampling.direction_to_light));
                    }
                }

                auto result = si.bsdf->sampleBSDF(si, rng);

                if(result.sample_probability > 0.0f)
                {
                    next_origin = si.position;
                    next_dir    = result.out_dir;
                    throughput *= result.bsdf_weight;
                    next_ray_valid = true;
                }
            }
        }
        else
        {
            if(params.environment_emitter)
            {
                radiance += throughput * params.environment_emitter->evalLight(si);
            }
        }

        ray_origin = next_origin;
        ray_dir    = next_dir;
    }

    if(params.frame_counter > 0)
    {
        // Mix with previous subframes if present!
        const float a                        = 1.0f / static_cast<float>(params.frame_counter + 1);
        const glm::vec3 prev_output_radiance = params.accumulation_buffer[pixel_index];
        radiance                             = glm::lerp(prev_output_radiance, radiance, a);
    }

    params.accumulation_buffer[pixel_index] = radiance;

    glm::vec3 tone_mapped =
        glm::clamp(glm::pow(1.0f - glm::exp(-radiance), glm::vec3(1.0f / 2.2f)), glm::vec3(0), glm::vec3(1));

    params.output_image[pixel_index] = glm::u8vec4((uint8_t)(tone_mapped.x * 255.0f),
                                                   (uint8_t)(tone_mapped.y * 255.0f),
                                                   (uint8_t)(tone_mapped.z * 255.0f),
                                                   255);
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