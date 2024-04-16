#include <Renderer/BSDFModels.h>
#include <Core/CUDA.h>

namespace atcg
{

namespace detail
{

void convertToTextureObject(const torch::Tensor& texture_data,
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

PBRBSDF::PBRBSDF(const Material& material)
{
    auto diffuse_texture   = material.getDiffuseTexture()->getData(atcg::GPU);
    auto metallic_texture  = material.getMetallicTexture()->getData(atcg::GPU);
    auto roughness_texture = material.getRoughnessTexture()->getData(atcg::GPU);

    PBRBSDFData data;

    detail::convertToTextureObject(diffuse_texture, _diffuse_texture, data.diffuse_texture);
    detail::convertToTextureObject(metallic_texture, _metallic_texture, data.metallic_texture);
    detail::convertToTextureObject(roughness_texture, _roughness_texture, data.roughness_texture);

    _bsdf_data_buffer.upload(&data);
}

PBRBSDF::~PBRBSDF()
{
    PBRBSDFData data;

    _bsdf_data_buffer.download(&data);

    CUDA_SAFE_CALL(cudaDestroyTextureObject(data.diffuse_texture));
    CUDA_SAFE_CALL(cudaDestroyTextureObject(data.metallic_texture));
    CUDA_SAFE_CALL(cudaDestroyTextureObject(data.roughness_texture));

    CUDA_SAFE_CALL(cudaFreeArray(_diffuse_texture));
    CUDA_SAFE_CALL(cudaFreeArray(_metallic_texture));
    CUDA_SAFE_CALL(cudaFreeArray(_roughness_texture));
}

void PBRBSDF::initializeBSDF(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                             const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_bsdf_filename = "./build/ptxmodules.dir/Debug/bsdf.ptx";
    auto prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_bsdf"});
    uint32_t idx    = sbt->addCallableEntry(prog_group, _bsdf_data_buffer.get());

    BSDFVPtrTable table;
    table.sampleCallIndex = idx;

    _vptr_table.upload(&table);
}

RefractiveBSDF::RefractiveBSDF(const Material& material)
{
    auto diffuse_texture = material.getDiffuseTexture()->getData(atcg::GPU);

    RefractiveBSDFData data;

    detail::convertToTextureObject(diffuse_texture, _diffuse_texture, data.diffuse_texture);
    data.ior = material.ior;

    _bsdf_data_buffer.upload(&data);
}

RefractiveBSDF::~RefractiveBSDF()
{
    RefractiveBSDFData data;

    _bsdf_data_buffer.download(&data);

    CUDA_SAFE_CALL(cudaDestroyTextureObject(data.diffuse_texture));

    CUDA_SAFE_CALL(cudaFreeArray(_diffuse_texture));
}

void RefractiveBSDF::initializeBSDF(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_bsdf_filename = "./build/ptxmodules.dir/Debug/bsdf.ptx";
    auto prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_refractivebsdf"});
    uint32_t idx    = sbt->addCallableEntry(prog_group, _bsdf_data_buffer.get());

    BSDFVPtrTable table;
    table.sampleCallIndex = idx;

    _vptr_table.upload(&table);
}
}    // namespace atcg