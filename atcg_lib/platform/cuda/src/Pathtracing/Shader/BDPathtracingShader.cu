#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Pathtracing/Shader/BDPathtracingShader.cuh>

#include <Math/Random.h>
#include <Pathtracing/SurfaceInteraction.h>
#include <Pathtracing/Payload.h>

extern "C"
{
    __constant__ BDPathtracingParams params;
}

#define CAMERA_PATH_LENGTH 10    // 0 - eye camera 1-14 - scatter events
#define LIGHT_PATH_LENGTH  10

struct PathVertex
{
    atcg::SurfaceInteraction si;
    glm::vec3 throughput = glm::vec3(1.0f);
};

__device__ inline void generateSubPath(PathVertex path[CAMERA_PATH_LENGTH],
                                       const uint32_t size,
                                       const glm::vec3& ray_origin_,
                                       const glm::vec3& ray_dir_,
                                       atcg::PCG32& rng)
{
    glm::vec3 ray_origin = ray_origin_;
    glm::vec3 ray_dir    = ray_dir_;

    bool next_ray_valid = true;

    glm::vec3 next_origin;
    glm::vec3 next_dir;

    glm::vec3 throughput = path[0].throughput;

    for(int n = 1; n < size; ++n)
    {
        if(!next_ray_valid) break;
        next_ray_valid = false;

        atcg::SurfaceInteraction si;
        traceWithDataPointer<atcg::SurfaceInteraction>(params.handle,
                                                       ray_origin,
                                                       ray_dir,
                                                       0.001f,
                                                       1e16f,
                                                       &si,
                                                       params.surface_trace_params);

        if(!si.valid)
        {
            next_ray_valid = false;
            break;
        }

        if(!si.bsdf)
        {
            next_ray_valid = false;
            break;
        }

        PathVertex vertex;
        vertex.si         = si;
        vertex.throughput = throughput;
        path[n]           = vertex;

        // PBR Sampling
        auto bsdf_sampling_result = si.bsdf->sampleBSDF(si, rng);

        if(bsdf_sampling_result.sample_probability > 0.0f)
        {
            next_origin = si.position;
            next_dir    = bsdf_sampling_result.out_dir;
            throughput *= bsdf_sampling_result.bsdf_weight;
            next_ray_valid = true;
            ray_origin     = next_origin;
            ray_dir        = next_dir;
        }
    }
}

__device__ inline void
generateCameraPath(const uint3& launch_idx, PathVertex path[CAMERA_PATH_LENGTH], atcg::PCG32& rng)
{
    glm::vec2 jitter = rng.next2d();
    float u          = (((float)launch_idx.x + jitter.x) / (float)params.image_width - 0.5f) * 2.0f;
    float v = (((float)(params.image_height - launch_idx.y) + jitter.y) / (float)params.image_height - 0.5f) * 2.0f;

    glm::vec3 cam_eye = glm::make_vec3(params.cam_eye);
    glm::vec3 U       = glm::make_vec3(params.U) * (float)params.image_width / (float)params.image_height;
    glm::vec3 V       = glm::make_vec3(params.V);
    glm::vec3 W       = glm::make_vec3(params.W) / glm::tan(glm::radians(params.fov_y / 2.0f));

    glm::vec3 ray_dir    = glm::normalize(u * U + v * V + W);
    glm::vec3 ray_origin = cam_eye;

    atcg::SurfaceInteraction camera_si;
    camera_si.position = ray_origin;
    camera_si.valid    = false;
    path[0].si         = camera_si;
    path[0].throughput = glm::vec3(1);

    generateSubPath(path, CAMERA_PATH_LENGTH, ray_origin, ray_dir, rng);
}

__device__ inline void generateLightPath(PathVertex path[LIGHT_PATH_LENGTH], atcg::PCG32& rng)
{
    uint32_t light_index = glm::clamp((uint32_t)(rng.nextFloat() * params.num_emitters), 0u, params.num_emitters - 1);
    float light_pdf      = 1.0f / (float)params.num_emitters;

    const atcg::EmitterVPtrTable* emitter = params.emitters[light_index];

    auto result          = emitter->samplePhoton(rng);
    glm::vec3 ray_origin = result.position;
    glm::vec3 ray_dir    = result.direction;
    result.pdf *= light_pdf;

    atcg::SurfaceInteraction light_si;
    light_si.position  = ray_origin;
    light_si.valid     = true;
    light_si.emitter   = emitter;
    light_si.normal    = result.normal;
    path[0].si         = light_si;
    path[0].throughput = result.radiance_weight / light_pdf / glm::pi<float>();

    generateSubPath(path, LIGHT_PATH_LENGTH, ray_origin, ray_dir, rng);
}

extern "C" __global__ void __raygen__rg()
{
    uint3 launch_idx = optixGetLaunchIndex();

    uint32_t pixel_index = launch_idx.x + params.image_width * launch_idx.y;
    uint64_t seed        = atcg::sampleTEA64(pixel_index, params.frame_counter);
    atcg::PCG32 rng(seed);

    glm::vec3 radiance(0);

    PathVertex camera_path[CAMERA_PATH_LENGTH];
    PathVertex light_path[LIGHT_PATH_LENGTH];

    generateCameraPath(launch_idx, camera_path, rng);
    generateLightPath(light_path, rng);

    uint32_t num_samples = 0;
    // t = 0, zero eye subpaths -> light directly hits sensor. We do not consider these paths here
    // t = 1, a light path is connected to the eye path and has contribution to another pixel, we do not handle this
    // here (for now)
    for(uint32_t t = 2; t < CAMERA_PATH_LENGTH; ++t)
    {
        for(uint32_t s = 0; s < LIGHT_PATH_LENGTH; ++s)
        {
            if(s == 0)
            {
                // No light vertices, camera path randomly intersects light source
                PathVertex camera_vertex = camera_path[t - 1];

                if(!camera_vertex.si.valid) continue;

                if(camera_vertex.si.emitter)
                {
                    glm::vec3 Le = camera_vertex.si.emitter->evalLight(camera_vertex.si);

                    radiance += Le * camera_vertex.throughput;
                }
            }
            else if(s == 1)
            {
                // TODO
            }
            else
            {
                PathVertex light_vertex  = light_path[s - 1];
                PathVertex camera_vertex = camera_path[t - 1];

                // End of path either one of these vertices was not samples
                if(!light_vertex.si.valid || !camera_vertex.si.valid)
                {
                    continue;
                }

                glm::vec3 dir_to_light          = light_vertex.si.position - camera_vertex.si.position;
                float distance_to_light_squared = glm::length2(dir_to_light);
                float distance_to_light         = glm::sqrt(distance_to_light_squared);
                dir_to_light /= distance_to_light;
                glm::vec3 dir_to_camera = -dir_to_light;

                float NdotL_camera = glm::max(0.0f, glm::dot(camera_vertex.si.normal, dir_to_light));
                float NdotV_light  = glm::max(0.0f, glm::dot(light_vertex.si.normal, dir_to_camera));

                // Points do not see each other
                if(NdotL_camera <= 0.0f || NdotV_light <= 0.0f)
                {
                    continue;
                }

                // Vertices are occluded
                if(traceOcclusion(params.handle,
                                  camera_vertex.si.position,
                                  dir_to_light,
                                  1e-3f,
                                  distance_to_light - 1e-3f,
                                  params.occlusion_trace_params))
                {
                    continue;
                }

                // Connect
                glm::vec3 alpha_L = light_vertex.throughput;
                glm::vec3 alpha_E = camera_vertex.throughput;

                float G                          = NdotL_camera * NdotV_light / distance_to_light_squared;
                atcg::BSDFEvalResult bsdf_light  = light_vertex.si.bsdf->evalBSDF(light_vertex.si, dir_to_camera);
                atcg::BSDFEvalResult bsdf_camera = camera_vertex.si.bsdf->evalBSDF(camera_vertex.si, dir_to_light);
                glm::vec3 c_st                   = bsdf_camera.bsdf_value * G * bsdf_light.bsdf_value;

                // TODO: weighting
                radiance += alpha_L * c_st * alpha_E;
            }

            // PathVertex light_vertex  = light_path[s];
            // PathVertex camera_vertex = camera_path[t];

            // glm::vec3 light_direction = glm::normalize(light_vertex.si.position - camera_vertex.si.position);
            // float light_distance      = glm::length(light_vertex.si.position - camera_vertex.si.position);

            // float NdotL_camera = glm::max(0.0f, glm::dot(light_direction, camera_vertex.si.normal));
            // float NdotV_light  = glm::max(0.0f, -glm::dot(light_direction, light_vertex.si.normal));

            // if(NdotL_camera <= 0.0f || NdotV_light <= 0.0f) continue;
            // // radiance += glm::vec3(1);

            // if(!camera_vertex.si.valid || !light_vertex.si.valid) continue;

            // if(traceOcclusion(params.handle,
            //                   camera_vertex.si.position,
            //                   light_direction,
            //                   1e-3f,
            //                   light_distance - 1e-3f,
            //                   params.occlusion_trace_params))
            // {
            //     continue;
            // }
        }
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

extern "C" __global__ void __miss__occlusion()
{
    setOcclusionPayload(false);
}