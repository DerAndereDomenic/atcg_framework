#pragma once


#ifdef ATCG_CUDA_BACKEND
    #include <iostream>
    #include "cuda_runtime.h"
    #include "device_launch_parameters.h"
namespace atcg
{
inline void check(cudaError_t error, char const* const func, const char* const file, int const line)
{
    if(error != cudaSuccess)
    {
        fprintf(stderr,
                "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file,
                line,
                static_cast<unsigned int>(error),
                cudaGetErrorString(error),
                func);
    }
}

constexpr bool cuda_available()
{
    return true;
}
}    // namespace atcg

    #ifdef NDEBUG
        #define CUDA_SAFE_CALL(val) val
    #else
        #define CUDA_SAFE_CALL(val) atcg::check((val), #val, __FILE__, __LINE__)
    #endif

    #define SET_DEVICE(dev)      CUDA_SAFE_CALL(cudaSetDevice(dev))
    #define SET_DEFAULT_DEVICE() SET_DEVICE(0)

    #define SYNCHRONIZE_DEFAULT_STREAM() CUDA_SAFE_CALL(cudaDeviceSynchronize())
    #define ATCG_HOST                    __host__
    #define ATCG_DEVICE                  __device__
    #define ATCG_HOST_DEVICE             __host__ __device__
    #define ATCG_GLOBAL                  __global__
#else
namespace atcg
{
constexpr bool cuda_available()
{
    return false;
}
}    // namespace atcg

    #define ATCG_HOST
    #define ATCG_DEVICE
    #define ATCG_HOST_DEVICE
    #define ATCG_GLOBAL

#endif