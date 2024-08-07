#pragma once

#include <optix.h>
#include <Core/Platform.h>
#include <Core/CUDA.h>

struct TraceParameters
{
    uint32_t rayFlags;
    uint32_t SBToffset;
    uint32_t SBTstride;
    uint32_t missSBTIndex;
};

#ifdef __CUDACC__

/**
 * @brief Create a pointer from two integer (payload)
 *
 * @param i0 First part of address
 * @param i1 Second part of address
 *
 * @return 64 bit pointer
 */
ATCG_INLINE ATCG_DEVICE void* unpackPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr           = reinterpret_cast<void*>(uptr);
    return ptr;
}

/**
 * @brief Pack a pointer into two int representation
 *
 * @param ptr The memory address to pack
 * @param i0 The frist part of the address
 * @param i1 The second part of the address
 */
ATCG_INLINE ATCG_DEVICE void packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0                  = uptr >> 32;
    i1                  = uptr & 0x00000000ffffffff;
}

/**
 * @brief Get the optix payload pointer of the current ray
 *
 * @tparam T The datatype the memory points to
 *
 * @return Pointer to the payload
 */
template<typename T>
ATCG_INLINE ATCG_DEVICE T* getPayloadDataPointer()
{
    // Get the pointer to the payload data
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
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

/**
 * @brief Set the occlusion payload
 *
 * @param occluded If the ray is occluded
 */
ATCG_INLINE ATCG_DEVICE void setOcclusionPayload(bool occluded)
{
    // Set the payload that _this_ ray will yield
    optixSetPayload_0(static_cast<uint32_t>(occluded));
}

/**
 * @brief Get the occlusion payload
 *
 * @return If the ray is occluded
 */
ATCG_INLINE ATCG_DEVICE bool getOcclusionPayload()
{
    // Get the payload that _this_ ray will yield
    return static_cast<bool>(optixGetPayload_0());
}

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

#endif
