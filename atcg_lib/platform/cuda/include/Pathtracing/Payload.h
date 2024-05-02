#pragma once

#include <optix.h>

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
