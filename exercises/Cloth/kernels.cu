#include "kernels.h"

namespace detail
{
__global__ void simulate(glm::vec3* points, float time, uint32_t n_points)
{
    size_t tid = atcg::threadIndex();
    if(tid >= n_points) return;

    int j = tid % n_points;
    int i = tid / n_points;

    points[tid].z = glm::sin(2.0f * glm::pi<float>() * (time) + j / 3.0f + i);
}
}    // namespace detail

void simulate(glm::vec3* points, uint32_t size, float time)
{
    size_t threads = 128;
    size_t blocks  = atcg::configure(size);
    detail::simulate<<<blocks, threads>>>(points, time, size);
    SYNCHRONIZE_DEFAULT_STREAM();
}