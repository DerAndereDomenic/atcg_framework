#pragma once

#include <Core/Platform.h>
#include <Core/CUDA.h>

namespace atcg
{
// Fix for global memory address atomic add (needed for atomicAdd in backward pass)
ATCG_DEVICE ATCG_INLINE void globalAtomicAdd(float* addr, float val)
{
    unsigned long long ptr = (unsigned long long)__cvta_generic_to_global(addr);
    asm volatile("atom.global.add.f32 %0, [%1], %2;" : "=f"(val) : "l"(ptr), "f"(val) : "memory");
}
}    // namespace atcg