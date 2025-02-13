#include <Emitter/EnvironmentEmitter.h>

namespace atcg
{

namespace detail
{

ATCG_INLINE static void convertToTextureObject(const torch::Tensor& texture_data,
                                               cudaArray_t& output_array,
                                               cudaTextureObject_t& output_texture)
{
    torch::Tensor data = texture_data;
    if(texture_data.size(2) == 2)
    {
        data = torch::cat({texture_data, torch::zeros_like(texture_data)}, -1);
    }

    if(texture_data.size(2) == 3)
    {
        data =
            torch::cat({texture_data,
                        torch::zeros_like(
                            texture_data.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}).unsqueeze(-1))},
                       -1);
    }


    uint32_t num_channels = data.size(2);
    bool isFloat          = data.dtype() == torch::kFloat32;

    uint32_t component_size = isFloat ? 32 : 8;

    cudaChannelFormatDesc format = {};
    format.f                     = isFloat ? cudaChannelFormatKindFloat : cudaChannelFormatKindUnsigned;
    format.x                     = component_size;
    format.y                     = num_channels == 4 ? component_size : 0;
    format.z                     = num_channels == 4 ? component_size : 0;
    format.w                     = num_channels == 4 ? component_size : 0;

    cudaExtent extent = {};
    extent.width      = data.size(1);
    extent.height     = data.size(0);
    extent.depth      = 0;

    CUDA_SAFE_CALL(cudaMalloc3DArray(&output_array, &format, extent));

    CUDA_SAFE_CALL(cudaMemcpy2DToArray(output_array,
                                       0,
                                       0,
                                       data.contiguous().data_ptr(),
                                       data.size(1) * data.size(2) * data.element_size(),
                                       data.size(1) * data.size(2) * data.element_size(),
                                       data.size(0),
                                       cudaMemcpyDeviceToDevice));

    cudaTextureDesc desc  = {};
    desc.normalizedCoords = 1;
    desc.addressMode[0]   = cudaAddressModeClamp;
    desc.addressMode[1]   = cudaAddressModeClamp;
    desc.addressMode[2]   = cudaAddressModeClamp;
    desc.readMode         = isFloat ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
    desc.filterMode       = cudaFilterModeLinear;

    cudaResourceDesc resDesc = {};
    resDesc.resType          = cudaResourceTypeArray;
    resDesc.res.array.array  = output_array;

    CUDA_SAFE_CALL(cudaCreateTextureObject(&output_texture, &resDesc, &desc, nullptr));
}
}    // namespace detail

EnvironmentEmitter::EnvironmentEmitter(const atcg::ref_ptr<atcg::Texture2D>& texture)
{
    _flags                   = EmitterFlags::DistantEmitter;
    auto environment_texture = texture->getData(atcg::GPU);

    EnvironmentEmitterData data;

    detail::convertToTextureObject(environment_texture, _environment_texture, data.environment_texture);

    _environment_emitter_data.upload(&data);
}

EnvironmentEmitter::~EnvironmentEmitter()
{
    EnvironmentEmitterData data;

    _environment_emitter_data.download(&data);

    CUDA_SAFE_CALL(cudaDestroyTextureObject(data.environment_texture));

    CUDA_SAFE_CALL(cudaFreeArray(_environment_texture));
}

void EnvironmentEmitter::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                            const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_emitter_filename = "./bin/EnvironmentEmitter_ptx.ptx";
    auto sample_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__sample_environmentemitter"});
    auto eval_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__eval_environmentemitter"});
    auto evalpdf_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__evalpdf_environmentemitter"});
    uint32_t sample_idx  = sbt->addCallableEntry(sample_prog_group, _environment_emitter_data.get());
    uint32_t eval_idx    = sbt->addCallableEntry(eval_prog_group, _environment_emitter_data.get());
    uint32_t evalpdf_idx = sbt->addCallableEntry(evalpdf_prog_group, _environment_emitter_data.get());

    EmitterVPtrTable table;
    table.flags            = _flags;
    table.sampleCallIndex  = sample_idx;
    table.evalCallIndex    = eval_idx;
    table.evalPdfCallIndex = evalpdf_idx;

    _vptr_table.upload(&table);
}
}    // namespace atcg