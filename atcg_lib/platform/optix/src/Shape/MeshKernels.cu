#include <Shape/MeshKernels.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <Core/Common.h>

namespace atcg
{

namespace detail
{
__global__ void
computeMeshTrianglePDFKernel(const torch::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> positions,
                             const torch::PackedTensorAccessor32<uint32_t, 2, at::RestrictPtrTraits> indices,
                             const glm::mat4 transform,
                             torch::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> pdf)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for(auto tid = id; tid < indices.size(0); tid += num_threads)
    {
        if(tid >= indices.size(0)) return;

        glm::u32vec3 triangle_indices = glm::u32vec3(indices[tid][0], indices[tid][1], indices[tid][2]);
        glm::vec3 local_P0            = glm::vec3(positions[triangle_indices.x][0],
                                                  positions[triangle_indices.x][1],
                                                  positions[triangle_indices.x][2]);
        glm::vec3 local_P1            = glm::vec3(positions[triangle_indices.y][0],
                                                  positions[triangle_indices.y][1],
                                                  positions[triangle_indices.y][2]);
        glm::vec3 local_P2            = glm::vec3(positions[triangle_indices.z][0],
                                                  positions[triangle_indices.z][1],
                                                  positions[triangle_indices.z][2]);

        glm::vec3 P0 = glm::vec3(transform * glm::vec4(local_P0, 1));
        glm::vec3 P1 = glm::vec3(transform * glm::vec4(local_P1, 1));
        glm::vec3 P2 = glm::vec3(transform * glm::vec4(local_P2, 1));

        // Compute triangle area
        float parallelogram_area = glm::length(glm::cross(P1 - P0, P2 - P0));
        float triangle_area      = 0.5f * parallelogram_area;

        // Write unnormalized pdf
        pdf[tid] = triangle_area;
    }
}

__global__ void
computeMeshEdgePDFKernel(const torch::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> positions,
                         const torch::PackedTensorAccessor32<uint32_t, 2, at::RestrictPtrTraits> indices,
                         const glm::mat4 transform,
                         torch::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> pdf)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for(auto tid = id; tid < indices.size(0); tid += num_threads)
    {
        if(tid >= indices.size(0)) return;

        glm::u32vec2 triangle_indices = glm::u32vec2(indices[tid][0], indices[tid][1]);
        glm::vec3 local_P0            = glm::vec3(positions[triangle_indices.x][0],
                                                  positions[triangle_indices.x][1],
                                                  positions[triangle_indices.x][2]);
        glm::vec3 local_P1            = glm::vec3(positions[triangle_indices.y][0],
                                                  positions[triangle_indices.y][1],
                                                  positions[triangle_indices.y][2]);

        glm::vec3 P0 = glm::vec3(transform * glm::vec4(local_P0, 1));
        glm::vec3 P1 = glm::vec3(transform * glm::vec4(local_P1, 1));

        // Compute triangle area
        float edge_length = glm::length(P1 - P0);

        // Write unnormalized pdf
        pdf[tid] = edge_length;
    }
}
}    // namespace detail


torch::Tensor
computeMeshTriangleAreas(const torch::Tensor& positions, const torch::Tensor& indices, const glm::mat4& transform)
{
    auto device = positions.device();

    at::cuda::CUDAGuard device_guard {device};

    torch::Tensor pdf = torch::zeros({indices.size(0)}, atcg::TensorOptions::floatDeviceOptions());
    const auto stream = at::cuda::getCurrentCUDAStream();

    const int threads_per_block = 128;
    dim3 grid;
    at::cuda::getApplyGrid(indices.size(0), grid, device.index(), threads_per_block);
    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

    detail::computeMeshTrianglePDFKernel<<<grid, threads, 0, stream>>>(
        positions.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        indices.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        transform,
        pdf.packed_accessor32<float, 1, torch::RestrictPtrTraits>());

    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    return pdf;
}

torch::Tensor
computeMeshEdgeLengths(const torch::Tensor& positions, const torch::Tensor& edges, const glm::mat4& transform)
{
    auto device = positions.device();

    at::cuda::CUDAGuard device_guard {device};

    torch::Tensor pdf = torch::zeros({edges.size(0)}, atcg::TensorOptions::floatDeviceOptions());
    const auto stream = at::cuda::getCurrentCUDAStream();

    const int threads_per_block = 128;
    dim3 grid;
    at::cuda::getApplyGrid(edges.size(0), grid, device.index(), threads_per_block);
    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

    detail::computeMeshEdgePDFKernel<<<grid, threads, 0, stream>>>(
        positions.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        edges.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
        transform,
        pdf.packed_accessor32<float, 1, torch::RestrictPtrTraits>());

    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    return pdf;
}

}    // namespace atcg