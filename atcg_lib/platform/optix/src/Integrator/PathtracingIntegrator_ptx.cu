#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Integrator/PathtracingData.cuh>

#include <Core/SurfaceInteraction.h>
#include <Core/Payload.h>

extern "C"
{
    __constant__ atcg::PathtracingParams params;
}

extern "C" __global__ void __raygen__rg()
{
    uint3 launch_idx = optixGetLaunchIndex();

    uint32_t pixel_index = launch_idx.x + params.image_width * launch_idx.y;

    float u = (((float)launch_idx.x) / (float)params.image_width - 0.5f) * 2.0f;
    float v = (((float)(params.image_height - launch_idx.y)) / (float)params.image_height - 0.5f) * 2.0f;

    glm::vec3 cam_eye = glm::make_vec3(params.cam_eye);
    glm::vec3 U       = glm::make_vec3(params.U) * (float)params.image_width / (float)params.image_height;
    glm::vec3 V       = glm::make_vec3(params.V);
    glm::vec3 W       = glm::make_vec3(params.W) / glm::tan(glm::radians(params.fov_y / 2.0f));

    glm::vec3 ray_dir    = glm::normalize(u * U + v * V + W);
    glm::vec3 ray_origin = cam_eye;

    glm::vec3 output(1.0f, 0.0f, 1.0);

    params.output_image[pixel_index] =
        glm::u8vec4((uint8_t)(output.x * 255.0f), (uint8_t)(output.y * 255.0f), (uint8_t)(output.z * 255.0f), 255);
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