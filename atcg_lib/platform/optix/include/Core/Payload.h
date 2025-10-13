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

#endif