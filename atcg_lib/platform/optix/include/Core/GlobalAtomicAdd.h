#pragma once

#include <Core/Platform.h>
#include <Core/CUDA.h>

namespace atcg
{
// Fix for global memory address atomic add (needed for atomicAdd in backward pass)
ATCG_DEVICE ATCG_INLINE float globalAtomicAdd(float* addr, float val)
{
    float old = val;
#ifdef __CUDACC__
    unsigned long long ptr = (unsigned long long)__cvta_generic_to_global(addr);
    asm volatile("atom.global.add.f32 %0, [%1], %2;" : "=f"(old) : "l"(ptr), "f"(val) : "memory");
#endif
    return old;
}

ATCG_DEVICE ATCG_INLINE int globalAtomicAdd(int* addr, int val)
{
    int old = val;
#ifdef __CUDACC__
    unsigned long long ptr = (unsigned long long)__cvta_generic_to_global(addr);
    asm volatile("atom.global.add.s32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(val) : "memory");
#endif
    return old;
}
}    // namespace atcg
