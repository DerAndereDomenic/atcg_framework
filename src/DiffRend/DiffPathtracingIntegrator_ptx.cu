#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include "DiffPathtracingData.cuh"

#include <Core/TraceParameters.h>
#include <Core/SurfaceInteraction.h>
#include <Core/Payload.h>
#include <Math/Random.h>

extern "C"
{
    __constant__ atcg::DiffPathtracingParams params;
}

struct RayContext
{
    bool valid;
    glm::vec3 origin;
    glm::vec3 direction;
    glm::vec3 throughput;
    glm::vec3 radiance;

    glm::vec3 delta_y;
};

extern "C" __global__ void __raygen__forward()
{
    uint3 launch_idx = optixGetLaunchIndex();

    uint32_t pixel_index = launch_idx.x + params.image_width * launch_idx.y;
    uint64_t seed        = atcg::sampleTEA64(pixel_index, params.rng_index);
    atcg::PCG32 rng(seed);

    glm::vec2 jitter = rng.next2d();
    float u          = (((float)launch_idx.x + jitter.x) / (float)params.image_width - 0.5f) * 2.0f;
    float v          = (((float)launch_idx.y + jitter.y) / (float)params.image_height - 0.5f) * 2.0f;

    glm::vec3 cam_eye = glm::make_vec3(params.cam_eye);
    glm::vec3 U       = glm::make_vec3(params.U) * (float)params.image_width / (float)params.image_height;
    glm::vec3 V       = glm::make_vec3(params.V);
    glm::vec3 W       = glm::make_vec3(params.W) / glm::tan(glm::radians(params.fov_y / 2.0f));

    RayContext ray;

    ray.direction     = glm::normalize(u * U + v * V + W);
    ray.origin        = cam_eye;
    ray.radiance      = glm::vec3(0);
    ray.throughput    = glm::vec3(1);
    ray.valid         = true;
    int32_t entity_id = -1;

    glm::vec3 next_origin;
    glm::vec3 next_dir;

    atcg::SurfaceInteraction last_si;
    float last_bsdf_pdf = 1.0f;

    for(int n = 0; n < 8; ++n)
    {
        if(!ray.valid) break;
        ray.valid = false;

        atcg::SurfaceInteraction si;
        atcg::traceWithDataPointer<atcg::SurfaceInteraction>(params.handle,
                                                             ray.origin,
                                                             ray.direction,
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
                ray.radiance += mis_weight * ray.throughput * si.emitter->evalLight(si);
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

                    atcg::EmitterSamplingResult emitter_sampling = emitter->sampleLight(si, rng);

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

                    atcg::BSDFEvalResult bsdf_result = si.bsdf->evalBSDF(si, emitter_sampling.direction_to_light);

                    float bsdf_pdf   = (int)(emitter->flags & atcg::EmitterFlags::InfinitesimalSize) != 0
                                           ? 0.0f
                                           : bsdf_result.sample_probability;
                    float mis_weight = emitter_sampling.sampling_pdf / (emitter_sampling.sampling_pdf + bsdf_pdf);

                    glm::vec3 radiance_nee = mis_weight * ray.throughput *
                                             emitter_sampling.radiance_weight_at_receiver * bsdf_result.bsdf_value *
                                             glm::abs(glm::dot(si.normal, emitter_sampling.direction_to_light));

                    ray.radiance += radiance_nee;
                } while(false);

                auto result = si.bsdf->sampleBSDF(si, rng);

                if(result.sample_probability > 0.0f)
                {
                    next_origin = si.position;
                    next_dir    = result.out_dir;
                    ray.throughput *= result.bsdf_weight;
                    ray.valid = true;

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
                ray.radiance += mis_weight * ray.throughput * params.environment_emitter->evalLight(si);
            }
        }

        ray.origin    = next_origin;
        ray.direction = next_dir;
    }

    params.current_sample[pixel_index] = ray.radiance;

    if(params.frame_counter > 0)
    {
        // Mix with previous subframes if present!
        const float a                        = 1.0f / static_cast<float>(params.frame_counter + 1);
        const glm::vec3 prev_output_radiance = params.accumulation_buffer[pixel_index];
        ray.radiance                         = glm::lerp(prev_output_radiance, ray.radiance, a);
    }

    params.accumulation_buffer[pixel_index] = ray.radiance;

    if(params.entity_ids)
    {
        params.entity_ids[pixel_index] = entity_id;
    }
}

extern "C" __global__ void __raygen__backward()
{
    uint3 launch_idx = optixGetLaunchIndex();

    uint32_t pixel_index = launch_idx.x + params.image_width * launch_idx.y;
    uint64_t seed        = atcg::sampleTEA64(pixel_index, params.rng_index);
    atcg::PCG32 rng(seed);

    glm::vec2 jitter = rng.next2d();
    float u          = (((float)launch_idx.x + jitter.x) / (float)params.image_width - 0.5f) * 2.0f;
    float v          = (((float)launch_idx.y + jitter.y) / (float)params.image_height - 0.5f) * 2.0f;

    glm::vec3 cam_eye = glm::make_vec3(params.cam_eye);
    glm::vec3 U       = glm::make_vec3(params.U) * (float)params.image_width / (float)params.image_height;
    glm::vec3 V       = glm::make_vec3(params.V);
    glm::vec3 W       = glm::make_vec3(params.W) / glm::tan(glm::radians(params.fov_y / 2.0f));

    RayContext ray;

    ray.direction     = glm::normalize(u * U + v * V + W);
    ray.origin        = cam_eye;
    ray.radiance      = params.accumulation_buffer[pixel_index];    // L from forward pass
    ray.throughput    = glm::vec3(1);
    ray.delta_y       = params.adjoint_y[pixel_index];
    ray.valid         = true;
    int32_t entity_id = -1;

    glm::vec3 next_origin;
    glm::vec3 next_dir;

    atcg::SurfaceInteraction last_si;
    float last_bsdf_pdf = 1.0f;

    for(int n = 0; n < 8; ++n)
    {
        if(!ray.valid) break;
        ray.valid = false;

        atcg::SurfaceInteraction si;
        atcg::traceWithDataPointer<atcg::SurfaceInteraction>(params.handle,
                                                             ray.origin,
                                                             ray.direction,
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
            // TODO: No NEE for now
            // Check for light source
            if(si.emitter)
            {
                bool mis_valid             = last_si.valid;
                float emitter_sampling_pdf = mis_valid ? si.emitter->evalLightSamplingPdf(last_si, si) : 0.0f;
                float mis_weight           = last_bsdf_pdf / (last_bsdf_pdf + emitter_sampling_pdf);
                // float mis_weight = 1.0f;
                ray.radiance -= mis_weight * ray.throughput * si.emitter->evalLight(si);
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

                    atcg::EmitterSamplingResult emitter_sampling = emitter->sampleLight(si, rng);

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

                    atcg::BSDFEvalResult bsdf_result = si.bsdf->evalBSDF(si, emitter_sampling.direction_to_light);

                    float bsdf_pdf   = (int)(emitter->flags & atcg::EmitterFlags::InfinitesimalSize) != 0
                                           ? 0.0f
                                           : bsdf_result.sample_probability;
                    float mis_weight = emitter_sampling.sampling_pdf / (emitter_sampling.sampling_pdf + bsdf_pdf);

                    glm::vec3 radiance_nee = mis_weight * ray.throughput *
                                             emitter_sampling.radiance_weight_at_receiver * bsdf_result.bsdf_value *
                                             glm::abs(glm::dot(si.normal, emitter_sampling.direction_to_light));

                    glm::vec3 grad_out = (ray.delta_y * (radiance_nee + 1e-4f)) / (bsdf_result.bsdf_value + 1e-4f);
                    si.bsdf->evalBSDFBackward(si, emitter_sampling.direction_to_light, grad_out);

                    ray.radiance -= radiance_nee;
                } while(false);

                auto result = si.bsdf->sampleBSDF(si, rng);

                if(result.sample_probability > 0.0f)
                {
                    // 𝛿𝜋 += backward_grad(bsdf_value, 𝛿𝐿 ∗ 𝐿 / bsdf_value)
                    // = 1/pi * dL * L / (albedo / pi) = dL * L / albedo
                    glm::vec3 grad_out = (ray.delta_y * (ray.radiance + 1e-4f)) /
                                         (result.bsdf_weight * result.sample_probability +
                                          1e-4f);    // bsdf_weight = bsdf_value * cos / p - so this should work?
                    si.bsdf->evalBSDFBackward(si, result.out_dir, grad_out);

                    next_origin = si.position;
                    next_dir    = result.out_dir;
                    ray.throughput *= result.bsdf_weight;
                    ray.valid = true;

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
                ray.radiance -= mis_weight * ray.throughput * params.environment_emitter->evalLight(si);
            }
        }

        ray.origin    = next_origin;
        ray.direction = next_dir;
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