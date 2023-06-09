#include "kernels.h"

#include <glm/gtx/transform.hpp>

namespace detail
{
__global__ void simulate(glm::vec3* points, float time, uint32_t n_points)
{
    uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid >= n_points) return;

    int j = tid % n_points;
    int i = tid / n_points;

    points[tid].z = glm::sin(2.0f * glm::pi<float>() * (time) + j / 3.0f + i);
}
}    // namespace detail

void simulate(glm::vec3* points, uint32_t size, float time)
{
    uint32_t threads = 128;
    uint32_t blocks  = glm::ceil((float)size / (float)threads);
    detail::simulate<<<blocks, threads>>>(points, time, size);
    cudaDeviceSynchronize();
}