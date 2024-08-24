#pragma cuda_source_property_format = PTX

#include <iostream>
#include <Math/Random.h>
#include <torch/types.h>
#include <Pathtracing/PathtracingShader.cuh>
#include <Pathtracing/SurfaceInteraction.h>
#include <Pathtracing/Stubs.h>
#include <Pathtracing/Tracing.h>
#include <Scene/Scene.h>
#include <Scene/Components.h>
#include <Scene/Entity.h>

extern "C"
{
    ATCG_RT_EXPORT PathtracingParams params;
}

template<typename T>
ATCG_INLINE void traceWithDataPointer(const atcg::IASAccelerationStructure* handle,
                                      const glm::vec3& ray_origin,
                                      const glm::vec3& ray_direction,
                                      float tmin,
                                      float tmax,
                                      T* payload_ptr)
{
    nanort::Ray<float> ray;
    memcpy(ray.org, glm::value_ptr(ray_origin), sizeof(glm::vec3));
    memcpy(ray.dir, glm::value_ptr(ray_direction), sizeof(glm::vec3));

    ray.min_t = tmin;
    ray.max_t = tmax;

    nanort::TriangleIntersector<> triangle_intersector(
        reinterpret_cast<const float*>(handle->getPositions().data_ptr()),
        reinterpret_cast<const uint32_t*>(handle->getFaces().data_ptr()),
        sizeof(float) * 3);

    nanort::TriangleIntersection<> isect;
    bool hit = handle->getBVH().Traverse(ray, triangle_intersector, &isect);

    if(!hit)
    {
        using funcT        = void (*)();
        auto [group, data] = atcg::g_sbt->sbt_entries_miss[0];    // TODO: TraceParams
        group->module->set_sbt(atcg::g_sbt);
        group->module->set_memory_for_function(std::get<0>(group->function).c_str(), data.data());
        group->module->set_payload_pointer((void*)payload_ptr);
        group->module->set_ray_origin(ray_origin);
        group->module->set_ray_direction(ray_direction);
        group->module->set_ray_tmin(tmin);
        group->module->set_ray_tmax(tmax);
        return reinterpret_cast<funcT>(std::get<1>(group->function))();
        return;
    }

    uint32_t mesh_id         = (uint32_t)handle->getMeshIDs()[(int)(isect.prim_id)].item<int>();
    glm::mat4 local_to_world = handle->getTransforms()[mesh_id];
    uint32_t offset          = handle->getOffsets()[mesh_id];

    using funcT        = void (*)();
    auto [group, data] = atcg::g_sbt->sbt_entries_hitgroup[mesh_id];
    group->module->set_sbt(atcg::g_sbt);
    group->module->set_memory_for_function(std::get<0>(group->function).c_str(), data.data());
    group->module->set_payload_pointer((void*)payload_ptr);
    group->module->set_ray_origin(ray_origin);
    group->module->set_ray_direction(ray_direction);
    group->module->set_ray_tmin(tmin);
    group->module->set_ray_tmax(tmax);
    group->module->set_primitive_idx(isect.prim_id - offset);
    group->module->set_local_to_world(local_to_world);
    group->module->set_barys(glm::vec2(isect.u, isect.v));
    return reinterpret_cast<funcT>(std::get<1>(group->function))();
}

extern "C" ATCG_RT_EXPORT void __miss__main()
{
    atcg::SurfaceInteraction* si = getPayloadDataPointer<atcg::SurfaceInteraction>();

    si->valid              = false;
    si->incoming_direction = getWorldRayDirection();
    si->incoming_distance  = std::numeric_limits<float>::infinity();
}

extern "C" ATCG_RT_EXPORT void __raygen__main()
{
    uint32_t x    = atcg::x;
    uint32_t y    = atcg::y;
    uint64_t seed = atcg::sampleTEA64(x + params.image_width * y, params.frame_counter);
    atcg::PCG32 rng(seed);

    glm::vec2 jitter = rng.next2d();
    float u          = (((float)x + jitter.x) / (float)params.image_width - 0.5f) * 2.0f;
    float v          = (((float)(params.image_height - y) + jitter.y) / (float)params.image_height - 0.5f) * 2.0f;

    glm::vec3 U = glm::make_vec3(params.U) * (float)params.image_width / (float)params.image_height;
    glm::vec3 V = glm::make_vec3(params.V);
    glm::vec3 W = glm::make_vec3(params.W) / glm::tan(glm::radians(params.fov_y / 2.0f));

    glm::vec3 ray_dir    = glm::normalize(u * U + v * V + W);
    glm::vec3 ray_origin = glm::make_vec3(params.cam_eye);
    glm::vec3 radiance(0);
    glm::vec3 throughput(1);

    glm::vec3 next_origin;
    glm::vec3 next_dir;

    atcg::SurfaceInteraction last_si;
    float last_bsdf_pdf = 1.0f;

    for(int n = 0; n < 10; ++n)
    {
        atcg::SurfaceInteraction si;
        traceWithDataPointer(params.handle, ray_origin, ray_dir, 1e-3f, 1e6f, &si);

        if(si.valid)
        {
            // PBR Sampling

            if(si.emitter)
            {
                bool mis_valid             = last_si.valid;
                float emitter_sampling_pdf = mis_valid ? si.emitter->evalLightSamplingPdf(last_si, si) : 0.0f;
                float mis_weight           = last_bsdf_pdf / (last_bsdf_pdf + emitter_sampling_pdf);
                radiance += mis_weight * throughput * si.emitter->evalLight(si);
            }

            if(si.bsdf)
            {
                // Next-event estimation
                for(uint32_t i = 0; i < params.num_emitters; ++i)
                {
                    auto emitter = params.emitters[i];
                    if(!emitter || si.emitter == emitter) continue;

                    atcg::EmitterSamplingResult emitter_sampling = emitter->sampleLight(si, rng);

                    if(emitter_sampling.sampling_pdf == 0) continue;

                    atcg::SurfaceInteraction si_dummy;
                    traceWithDataPointer(params.handle,
                                         si.position,
                                         emitter_sampling.direction_to_light,
                                         1e-3f,
                                         emitter_sampling.distance_to_light - 1e-3f,
                                         &si_dummy);

                    bool occluded = si_dummy.valid;
                    if(occluded)
                    {
                        continue;
                    }

                    atcg::BSDFEvalResult bsdf_result = si.bsdf->evalBSDF(si, emitter_sampling.direction_to_light);

                    float mis_weight = emitter_sampling.sampling_pdf /
                                       (emitter_sampling.sampling_pdf + bsdf_result.sample_probability);

                    radiance += mis_weight * throughput * emitter_sampling.radiance_weight_at_receiver *
                                bsdf_result.bsdf_value *
                                glm::max(0.0f, glm::dot(si.normal, emitter_sampling.direction_to_light));
                }

                atcg::BSDFSamplingResult result = si.bsdf->sampleBSDF(si, rng);
                if(result.sample_probability > 0.0f)
                {
                    next_origin = si.position;
                    next_dir    = result.out_dir;
                    throughput *= result.bsdf_weight;


                    last_si       = si;
                    last_bsdf_pdf = result.sample_probability;

                    if((int)(si.bsdf->flags & atcg::BSDFComponentType::AnyDelta) != 0)
                    {
                        last_si.valid = false;
                    }
                }
                else
                {
                    break;
                }
            }
        }
        else
        {
            // Can't hit environment emitter via bsdf sampling
            if(params.environment_emitter) radiance += throughput * params.environment_emitter->evalLight(si);

            break;
        }

        ray_origin = next_origin;
        ray_dir    = next_dir;
    }

    if(params.frame_counter > 0)
    {
        // Mix with previous subframes if present!
        const float a                        = 1.0f / static_cast<float>(params.frame_counter + 1);
        const glm::vec3 prev_output_radiance = (params.accumulation_buffer)[x + params.image_width * y];
        radiance                             = glm::lerp(prev_output_radiance, radiance, a);
    }

    params.accumulation_buffer[x + params.image_width * y] = radiance;

    glm::vec3 tone_mapped =
        glm::clamp(glm::pow(1.0f - glm::exp(-radiance), glm::vec3(1.0f / 2.2f)), glm::vec3(0), glm::vec3(1));

    params.output_image[x + params.image_width * y] = glm::u8vec4((uint8_t)(tone_mapped.x * 255.0f),
                                                                  (uint8_t)(tone_mapped.y * 255.0f),
                                                                  (uint8_t)(tone_mapped.z * 255.0f),
                                                                  255);
}