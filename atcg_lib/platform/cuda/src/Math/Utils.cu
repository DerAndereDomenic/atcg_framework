#include <Math/Utils.h>

namespace atcg
{

namespace detail
{
__device__ float atomicMax(float* address, float val)
{
    int* address_as_i = (int*)address;
    int old           = *address_as_i, assumed;
    do {
        assumed = old;
        old     = ::atomicCAS(address_as_i, assumed, __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while(assumed != old);
    return __int_as_float(old);
}

ATCG_GLOBAL void mean_and_scale_kernel(Vertex* points, uint32_t num_points, glm::vec3* mean_point, float* max_scale)
{
    const std::size_t tid = atcg::threadIndex();
    if(tid >= num_points) { return; }

    atomicAdd(&(mean_point->x), points[tid].position.x);
    atomicAdd(&(mean_point->y), points[tid].position.y);
    atomicAdd(&(mean_point->z), points[tid].position.z);

    float local_max = -INFINITY;
    local_max       = glm::max(local_max, points[tid].position.x);
    local_max       = glm::max(local_max, points[tid].position.y);
    local_max       = glm::max(local_max, points[tid].position.z);

    atomicMax(max_scale, local_max);
}

ATCG_GLOBAL void
translate_and_scale_kernel(Vertex* points, uint32_t num_points, glm::vec3* mean_point, float* max_scale)
{
    const std::size_t tid = atcg::threadIndex();
    if(tid >= num_points) { return; }

    glm::vec3 m = *mean_point / static_cast<float>(num_points);

    points[tid].position = (points[tid].position - m) / (*max_scale);
}
}    // namespace detail

void normalize(const atcg::ref_ptr<Graph>& graph)
{
    uint32_t n_points = graph->n_vertices();

    std::size_t threads = 128;
    std::size_t blocks  = atcg::configure(n_points);

    atcg::ref_ptr<glm::vec3, atcg::device_allocator> mean_point(1);
    atcg::ref_ptr<float, atcg::device_allocator> max_scale(1);

    Vertex* vertices = (Vertex*)graph->getVerticesBuffer()->getDevicePointer();

    detail::mean_and_scale_kernel<<<blocks, threads>>>(vertices, n_points, mean_point.get(), max_scale.get());
    CUDA_SAFE_CALL(cudaGetLastError());
    SYNCHRONIZE_DEFAULT_STREAM();

    detail::translate_and_scale_kernel<<<blocks, threads>>>(vertices, n_points, mean_point.get(), max_scale.get());
    CUDA_SAFE_CALL(cudaGetLastError());
    SYNCHRONIZE_DEFAULT_STREAM();
}
}    // namespace atcg