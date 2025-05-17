#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

// #include <Integrator/PathtracingData.cuh>

#include <Core/TraceParameters.h>
#include <Math/Random.h>
#include <Core/SurfaceInteraction.h>
#include <Core/Payload.h>
#include "RadiosityParams.h"

extern "C"
{
    __constant__ RadiosityParams params;
}

extern "C" __global__ void __raygen__rg()
{
    uint3 launch_idx = optixGetLaunchIndex();

    if(launch_idx.x >= params.n_faces || launch_idx.y >= params.n_faces) return;

    if(launch_idx.x >= launch_idx.y) return;

    glm::u32vec3 tri_i = params.shape->faces[launch_idx.x];
    glm::u32vec3 tri_j = params.shape->faces[launch_idx.y];

    glm::vec3 p_i1 = params.shape->positions[tri_i.x];
    glm::vec3 p_i2 = params.shape->positions[tri_i.y];
    glm::vec3 p_i3 = params.shape->positions[tri_i.z];

    glm::vec3 p_j1 = params.shape->positions[tri_j.x];
    glm::vec3 p_j2 = params.shape->positions[tri_j.y];
    glm::vec3 p_j3 = params.shape->positions[tri_j.z];

    glm::vec3 n_i = glm::cross((p_i2 - p_i1), (p_i3 - p_i1));
    float area_i  = 0.5f * glm::length(n_i);

    glm::vec3 n_j = glm::cross((p_j2 - p_j1), (p_j3 - p_j1));
    float area_j  = 0.5f * glm::length(n_j);

    n_i /= (2.0f * area_i);
    n_j /= (2.0f * area_j);

    glm::vec3 c_i = (p_i1 + p_i2 + p_i3) / 3.0f;
    glm::vec3 c_j = (p_j1 + p_j2 + p_j3) / 3.0f;

    glm::vec3 ray_direction = c_j - c_i;
    float ray_distance      = glm::length(ray_direction);
    ray_direction /= ray_distance;

    float cos_theta_i = glm::max(0.0f, glm::dot(n_i, ray_direction));
    float cos_theta_j = glm::max(0.0f, -glm::dot(n_j, ray_direction));

    float G = 0.0f;
    if(cos_theta_i > 0 && cos_theta_j > 0)
    {
        bool occluded = atcg::traceOcclusion(params.handle,
                                             c_i,
                                             ray_direction,
                                             1e-4,
                                             ray_distance - 1e-4f,
                                             params.occlusion_trace_params);

        if(!occluded)
        {
            G = cos_theta_i * cos_theta_j / (ray_distance * ray_distance * glm::pi<float>());
        }
    }

    params.form_factors[launch_idx.x + params.n_faces * launch_idx.y] = G * area_j;
    params.form_factors[launch_idx.y + params.n_faces * launch_idx.x] = G * area_i;
}


extern "C" __global__ void __miss__occlusion()
{
    setOcclusionPayload(false);
}