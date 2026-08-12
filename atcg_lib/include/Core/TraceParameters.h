#pragma once

#include <Core/Payload.h>

namespace atcg
{
struct TraceParameters
{
    uint32_t rayFlags;
    uint32_t SBToffset;
    uint32_t SBTstride;
    uint32_t missSBTIndex;
};

#ifdef __CUDACC__
/**
 * @brief Trace a ray to check for occlusion
 *
 * @param handle Handle to the acceleration structure
 * @param ray_origin The ray origin
 * @param ray_direction The ray direction
 * @param tmin The minimum ray distance
 * @param tmax The maximum ray distance
 * @param trace_params The trace parameters
 *
 * @return If the ray is occluded (hits any geometry)
 */
ATCG_INLINE ATCG_DEVICE bool traceOcclusion(OptixTraversableHandle handle,
                                            const glm::vec3& ray_origin,
                                            const glm::vec3& ray_direction,
                                            float tmin,
                                            float tmax,
                                            TraceParameters trace_params)
{
    uint32_t occluded = 1u;
    float3 o          = make_float3(ray_origin.x, ray_origin.y, ray_origin.z);
    float3 d          = make_float3(ray_direction.x, ray_direction.y, ray_direction.z);
    optixTrace(handle,
               o,
               d,
               tmin,
               tmax,
               0.0f,                      // rayTime
               OptixVisibilityMask(1),    // visibilityMask
               trace_params.rayFlags,     // OPTIX_RAY_FLAG_NONE
               trace_params.SBToffset,
               trace_params.SBTstride,
               trace_params.missSBTIndex,
               occluded);
    return occluded != 0;
}

/**
 * @brief Trace a ray and write it into the payload
 *
 * @param handle Handle to the acceleration structure
 * @param ray_origin The ray origin
 * @param ray_direction The ray direction
 * @param tmin The minimum ray distance
 * @param tmax The maximum ray distance
 * @param payload_ptr The output payload
 * @param trace_params The trace parameters
 */
template<typename T>
ATCG_INLINE ATCG_DEVICE void traceWithDataPointer(OptixTraversableHandle handle,
                                                  const glm::vec3& ray_origin,
                                                  const glm::vec3& ray_direction,
                                                  float tmin,
                                                  float tmax,
                                                  T* payload_ptr,
                                                  const TraceParameters& trace_params)
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
               trace_params.rayFlags,     // OPTIX_RAY_FLAG_NONE
               trace_params.SBToffset,
               trace_params.SBTstride,
               trace_params.missSBTIndex,
               u0,     // payload 0
               u1);    // payload 1
    // optixTrace operation will have updated content of *payload_ptr
}
#endif
}    // namespace atcg