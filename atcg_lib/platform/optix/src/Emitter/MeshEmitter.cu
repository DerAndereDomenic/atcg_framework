#include <Emitter/MeshEmitter.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <Shape/MeshShape.h>
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

__global__ void computeMeshTriangleCDFKernel(torch::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> cdf)
{
    float acc = 0;
    for(uint32_t i = 0; i < cdf.size(0); ++i)
    {
        acc += cdf[i];
        cdf[i] = acc;
    }
}

__global__ void normalizeMeshTriangleCDFKernel(torch::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> cdf,
                                               float total_value)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for(auto tid = id; tid < cdf.size(0); tid += num_threads)
    {
        if(tid >= cdf.size(0)) return;

        cdf[tid] /= total_value;
    }
}
}    // namespace detail

MeshEmitter::MeshEmitter(const Dictionary& dict)
{
    auto shape            = dict.getValue<atcg::ref_ptr<MeshShape>>("shape");
    auto texture_emissive = dict.getValue<atcg::ref_ptr<Texture2D>>("texture_emissive");
    auto emission_scaling = dict.getValue<float>("emission_scaling");
    glm::mat4 transform   = dict.getValue<glm::mat4>("transform");

    MeshEmitterData data;

    atcg::convertToTextureObject(texture_emissive->getData(atcg::GPU), _emissive_texture, data.emissive_texture);

    data.emitter_scaling = emission_scaling;

    torch::Tensor positions = shape->getPositions();
    torch::Tensor uvs       = shape->getUVs();
    torch::Tensor faces     = shape->getFaces();

    data.positions = (glm::vec3*)positions.data_ptr();
    data.uvs       = (glm::vec3*)uvs.data_ptr();
    data.faces     = (glm::u32vec3*)faces.data_ptr();
    data.num_faces = faces.size(0);

    _mesh_cdf = torch::zeros({faces.size(0)}, atcg::TensorOptions::floatDeviceOptions());

    auto device = _mesh_cdf.device();

    {
        at::cuda::CUDAGuard device_guard {device};
        const auto stream = at::cuda::getCurrentCUDAStream();

        const int threads_per_block = 128;
        dim3 grid;
        at::cuda::getApplyGrid(faces.size(0), grid, device.index(), threads_per_block);
        dim3 threads = at::cuda::getApplyBlock(threads_per_block);

        detail::computeMeshTrianglePDFKernel<<<grid, threads, 0, stream>>>(
            positions.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            faces.packed_accessor32<uint32_t, 2, torch::RestrictPtrTraits>(),
            transform,
            _mesh_cdf.packed_accessor32<float, 1, torch::RestrictPtrTraits>());

        AT_CUDA_CHECK(cudaGetLastError());
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    {
        at::cuda::CUDAGuard device_guard {device};
        const auto stream = at::cuda::getCurrentCUDAStream();

        detail::computeMeshTriangleCDFKernel<<<1, 1, 0, stream>>>(
            _mesh_cdf.packed_accessor32<float, 1, torch::RestrictPtrTraits>());

        AT_CUDA_CHECK(cudaGetLastError());
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    data.total_area = _mesh_cdf.index({_mesh_cdf.size(0) - 1}).cpu().item<float>();
    {
        at::cuda::CUDAGuard device_guard {device};
        const auto stream = at::cuda::getCurrentCUDAStream();

        const int threads_per_block = 128;
        dim3 grid;
        at::cuda::getApplyGrid(_mesh_cdf.size(0), grid, device.index(), threads_per_block);
        dim3 threads = at::cuda::getApplyBlock(threads_per_block);

        detail::normalizeMeshTriangleCDFKernel<<<grid, threads, 0, stream>>>(
            _mesh_cdf.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            data.total_area);

        AT_CUDA_CHECK(cudaGetLastError());
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    data.mesh_cdf       = (float*)_mesh_cdf.data_ptr();
    data.local_to_world = transform;
    data.world_to_local = glm::inverse(transform);

    _mesh_emitter_data.upload(&data);
}

MeshEmitter::~MeshEmitter()
{
    MeshEmitterData data;

    _mesh_emitter_data.download(&data);

    CUDA_SAFE_CALL(cudaDestroyTextureObject(data.emissive_texture));

    CUDA_SAFE_CALL(cudaFreeArray(_emissive_texture));
}

void MeshEmitter::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                     const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_emitter_filename = "./bin/MeshEmitter_ptx.ptx";
    auto sample_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__sample_meshemitter"});
    auto eval_prog_group = pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__eval_meshemitter"});
    auto evalpdf_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__evalpdf_meshemitter"});
    uint32_t sample_idx   = sbt->addCallableEntry(sample_prog_group, _mesh_emitter_data.get());
    uint32_t eval_idx     = sbt->addCallableEntry(eval_prog_group, _mesh_emitter_data.get());
    uint32_t eval_pdf_idx = sbt->addCallableEntry(evalpdf_prog_group, _mesh_emitter_data.get());

    EmitterVPtrTable table;
    table.sampleCallIndex  = sample_idx;
    table.evalCallIndex    = eval_idx;
    table.evalPdfCallIndex = eval_pdf_idx;

    _vptr_table.upload(&table);
}
}    // namespace atcg