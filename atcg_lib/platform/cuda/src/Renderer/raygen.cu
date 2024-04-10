#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include "Params.cuh"

#include <Math/Random.h>
#include <Math/SurfaceInteraciton.h>

extern "C"
{
    __constant__ Params params;
}

inline __device__ void* unpackPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr           = reinterpret_cast<void*>(uptr);
    return ptr;
}


inline __device__ void packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0                  = uptr >> 32;
    i1                  = uptr & 0x00000000ffffffff;
}

template<typename T>
inline __device__ T* getPayloadDataPointer()
{
    // Get the pointer to the payload data
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

template<typename T>
inline __device__ void traceWithDataPointer(OptixTraversableHandle handle,
                                            glm::vec3 ray_origin,
                                            glm::vec3 ray_direction,
                                            float tmin,
                                            float tmax,
                                            T* payload_ptr)
{
    uint32_t u0, u1;
    packPointer(payload_ptr, u0, u1);
    float3 o = make_float3(ray_origin.x, ray_origin.y, ray_origin.z);
    float3 d = make_float3(ray_direction.x, ray_direction.y, ray_direction.z);
    optixTrace(handle,
               o,
               d,
               tmin,
               tmax,
               0.0f,                      // rayTime
               OptixVisibilityMask(1),    // visibilityMask
               OPTIX_RAY_FLAG_NONE,       // OPTIX_RAY_FLAG_NONE
               0,
               1,
               0,
               u0,     // payload 0
               u1);    // payload 1
    // optixTrace operation will have updated content of *payload_ptr
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
    glm::vec3 U       = glm::make_vec3(params.U);
    glm::vec3 V       = glm::make_vec3(params.V);
    glm::vec3 W       = glm::make_vec3(params.W);

    glm::vec3 ray_dir    = glm::normalize(u * U + v * V + W);
    glm::vec3 ray_origin = cam_eye;
    glm::vec3 radiance(0);
    glm::vec3 throughput(1);

    glm::vec3 next_origin;
    glm::vec3 next_dir;

    for(int n = 0; n < 10; ++n)
    {
        atcg::SurfaceInteraction si;
        traceWithDataPointer<atcg::SurfaceInteraction>(params.handle, ray_origin, ray_dir, 0.001f, 1e16f, &si);

        if(si.valid)
        {
            // PBR Sampling
            auto result = si.bsdf->sampleBSDF(si, rng);

            if(result.sample_probability > 0.0f)
            {
                next_origin = si.position;
                next_dir    = result.out_dir;
                throughput *= result.bsdf_weight;
            }
            else
            {
                break;
            }
        }
        else
        {
            if(params.hasSkybox)
            {
                float theta = std::acos(ray_dir.y) / glm::pi<float>();
                float phi   = (std::atan2(ray_dir.z, ray_dir.x) + glm::pi<float>()) / (2.0f * glm::pi<float>());

                glm::vec2 uv(phi, theta);

                glm::ivec2 pixel(uv.x * params.skybox_width, params.skybox_height - uv.y * params.skybox_height);
                pixel = glm::clamp(pixel, glm::ivec2(0), glm::ivec2(params.skybox_width - 1, params.skybox_height - 1));

                radiance = throughput * glm::vec3(params.skybox_data[pixel.x + params.skybox_width * pixel.y]);
            }

            break;
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

extern "C" __global__ void __closesthit__ch()
{
    atcg::SurfaceInteraction* si = getPayloadDataPointer<atcg::SurfaceInteraction>();
    const HitGroupData* sbt_data = *reinterpret_cast<const HitGroupData**>(optixGetSbtDataPointer());

    float3 optix_world_origin = optixGetWorldRayOrigin();
    float3 optix_world_dir    = optixGetWorldRayDirection();
    glm::vec3 ray_origin      = glm::make_vec3((float*)&optix_world_origin);
    glm::vec3 ray_dir         = glm::make_vec3((float*)&optix_world_dir);
    float tmax                = optixGetRayTmax();

    si->valid              = true;
    si->incoming_distance  = tmax;
    si->incoming_direction = ray_dir;
    si->primitive_idx      = optixGetPrimitiveIndex();
    float2 optix_barys     = optixGetTriangleBarycentrics();
    si->barys              = glm::make_vec2((float*)&optix_barys);

    glm::u32vec3 triangle = sbt_data->faces[si->primitive_idx];

    const glm::vec3 P0 = sbt_data->positions[triangle.x];
    const glm::vec3 P1 = sbt_data->positions[triangle.y];
    const glm::vec3 P2 = sbt_data->positions[triangle.z];
    si->position       = (1.0f - si->barys.x - si->barys.y) * P0 + si->barys.x * P1 + si->barys.y * P2;
    // Transform local position to world position
    float3 optix_pos =
        optixTransformPointFromObjectToWorldSpace(make_float3(si->position.x, si->position.y, si->position.z));
    si->position = glm::make_vec3((float*)&optix_pos);

    const glm::vec3 N0 = sbt_data->normals[triangle.x];
    const glm::vec3 N1 = sbt_data->normals[triangle.y];
    const glm::vec3 N2 = sbt_data->normals[triangle.z];
    si->normal         = (1.0f - si->barys.x - si->barys.y) * N0 + si->barys.x * N1 + si->barys.y * N2;
    // Transform local position to world position
    float3 optix_normal =
        optixTransformNormalFromObjectToWorldSpace(make_float3(si->normal.x, si->normal.y, si->normal.z));
    si->normal = glm::normalize(glm::make_vec3((float*)&optix_normal));

    const glm::vec3 UV0 = sbt_data->uvs[triangle.x];
    const glm::vec3 UV1 = sbt_data->uvs[triangle.y];
    const glm::vec3 UV2 = sbt_data->uvs[triangle.z];
    si->uv              = (1.0f - si->barys.x - si->barys.y) * UV0 + si->barys.x * UV1 + si->barys.y * UV2;

    si->bsdf = sbt_data->bsdf;
}

extern "C" __global__ void __miss__ms()
{
    atcg::SurfaceInteraction* si = getPayloadDataPointer<atcg::SurfaceInteraction>();

    si->valid = false;
}