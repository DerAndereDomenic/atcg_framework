#pragma once


#ifdef ATCG_CUDA_BACKEND
    #include <iostream>
    #include "cuda_runtime.h"
    #include "device_launch_parameters.h"
    #include <Core/glm.h>
    #include <Math/Functions.h>
    #include <Core/Log.h>
namespace atcg
{
inline void check(cudaError_t error, char const* const func, const char* const file, int const line)
{
    if(error != cudaSuccess)
    {
        ATCG_ERROR("CUDA error at {0}:{1} code={2}({3}) \"{4}\" \n",
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

__device__ inline size_t threadIndex()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ inline glm::vec2 threadIndex2D()
{
    return glm::vec2(threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y * blockDim.y);
}

__device__ inline glm::vec3 threadIndex3D()
{
    return glm::vec3(threadIdx.x + blockIdx.x * blockDim.x,
                     threadIdx.y + blockIdx.y * blockDim.y,
                     threadIdx.z + blockIdx.z * blockDim.z);
}

inline size_t configure(size_t size, size_t n_threads = 128)
{
    return Math::ceil_div<size_t>(size, n_threads);
}

inline dim3 configure(glm::u32vec3 thread_count, glm::u32vec3 block_size)
{
    glm::u32vec3 block_count = Math::ceil_div<glm::u32vec3>(thread_count, block_size);
    return {block_count.x, block_count.y, block_count.z};
}

typedef cudaArray_t textureArray;
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
typedef void* textureArray;
}    // namespace atcg

    #define ATCG_HOST
    #define ATCG_DEVICE
    #define ATCG_HOST_DEVICE
    #define ATCG_GLOBAL


#endif