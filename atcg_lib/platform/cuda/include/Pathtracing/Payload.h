#pragma once

#include <optix.h>

struct TraceParameters
{
    uint32_t rayFlags;
    uint32_t SBToffset;
    uint32_t SBTstride;
    uint32_t missSBTIndex;
};

#ifdef __CUDACC__

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

__forceinline__ __device__ void setOcclusionPayload(bool occluded)
{
    // Set the payload that _this_ ray will yield
    optixSetPayload_0(static_cast<uint32_t>(occluded));
}

__forceinline__ __device__ bool getOcclusionPayload()
{
    // Get the payload that _this_ ray will yield
    return static_cast<bool>(optixGetPayload_0());
}

inline __device__ bool traceOcclusion(OptixTraversableHandle handle,
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
