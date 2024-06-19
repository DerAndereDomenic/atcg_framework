#pragma once


#include <Core/Platform.h>
#ifdef ATCG_CUDA_BACKEND
    #include "cuda_runtime.h"
    #include "device_launch_parameters.h"
    #include <Core/glm.h>
    #include <Core/Log.h>

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
namespace atcg
{
ATCG_INLINE void check(cudaError_t error, char const* const func, const char* const file, int const line)
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

ATCG_CONSTEXPR ATCG_FORCE_INLINE bool cuda_available()
{
    return true;
}

ATCG_DEVICE ATCG_FORCE_INLINE size_t threadIndex()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

ATCG_DEVICE ATCG_FORCE_INLINE glm::vec2 threadIndex2D()
{
    return glm::vec2(threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y * blockDim.y);
}

ATCG_DEVICE ATCG_FORCE_INLINE glm::vec3 threadIndex3D()
{
    return glm::vec3(threadIdx.x + blockIdx.x * blockDim.x,
                     threadIdx.y + blockIdx.y * blockDim.y,
                     threadIdx.z + blockIdx.z * blockDim.z);
}

template<typename T>
T ceil_div(T num, T den)
{
    return (num - T(1)) / den + T(1);
}

ATCG_INLINE size_t configure(size_t size, size_t n_threads = 128)
{
    return ceil_div<size_t>(size, n_threads);
}

ATCG_INLINE dim3 configure(glm::u32vec3 thread_count, glm::u32vec3 block_size)
{
    glm::u32vec3 block_count = ceil_div<glm::u32vec3>(thread_count, block_size);
    return {block_count.x, block_count.y, block_count.z};
}

typedef cudaArray_t textureArray;
typedef cudaTextureObject_t textureObject;
typedef cudaSurfaceObject_t surfaceObject;
}    // namespace atcg

#else
namespace atcg
{
constexpr bool cuda_available()
{
    return false;
}
typedef void* textureArray;
typedef void* textureObject;
typedef void* surfaceObject;
}    // namespace atcg

    #define ATCG_HOST
    #define ATCG_DEVICE
    #define ATCG_HOST_DEVICE
    #define ATCG_GLOBAL


#endif