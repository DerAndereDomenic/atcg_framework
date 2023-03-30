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

    #ifdef NDEBUG
        #define cudaSafeCall(val) val
    #else
        #define cudaSafeCall(val) check((val), #val, __FILE__, __LINE__)
    #endif

    #define setDevice(dev)     cudaSafeCall(cudaSetDevice(dev))
    #define setDefaultDevice() setDevice(0)

    #define synchronizeDefaultStream() cudaSafeCall(cudaDeviceSynchronize())

constexpr bool cuda_available()
{
    return true;
}

#else
namespace atcg
{
constexpr bool cuda_available()
{
    return false;
}

    #define ATCG_HOST        __host__
    #define ATCG_DEVICE      __device__
    #define ATCG_HOST_DEVICE __host__ __device__
    #define ATCG_GLOBAL      __global__

#endif
}    // namespace atcg