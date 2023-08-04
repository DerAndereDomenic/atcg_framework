#include "kernels.h"

namespace detail
{
__global__ void simulate(atcg::BufferView<glm::vec3> points, float time, uint32_t n_points)
{
    size_t tid = atcg::threadIndex();
    if(tid >= n_points) return;

    int j = tid % n_points;
    int i = tid / n_points;

    updatePoint(points, tid, j, i, time);
}
}    // namespace detail

void simulate(atcg::BufferView<glm::vec3>& points, uint32_t size, float time)
{
    size_t threads = 128;
    size_t blocks  = atcg::configure(size);
    detail::simulate<<<blocks, threads>>>(points, time, size);
    CUDA_SAFE_CALL(cudaGetLastError());
    SYNCHRONIZE_DEFAULT_STREAM();
}