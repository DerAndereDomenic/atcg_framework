#pragma once

#include <cstdint>

#ifdef __CUDACC__
    #define ATCG_INLINE       inline
    #define ATCG_FORCE_INLINE __forceinline__
    #define ATCG_CONSTEXPR
    #define ATCG_CONSTANT __constant__
    #define ATCG_CONST    const
#else
    #define ATCG_INLINE       inline
    #define ATCG_FORCE_INLINE ATCG_INLINE
    #define ATCG_CONSTEXPR    constexpr
    #define ATCG_CONSTANT
    #define ATCG_CONST const
#endif

namespace atcg
{
ATCG_INLINE ATCG_CONSTEXPR bool isLittleEndian()
{
    int num = 1;
    return (*(char*)&num) == char(1);
}
}    // namespace atcg