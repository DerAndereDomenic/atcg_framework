#include <Renderer/BSDFModels.h>
#include <Core/CUDA.h>

namespace atcg
{
PBRBSDF::PBRBSDF(const Material& material)
{
    _diffuse_texture   = material.getDiffuseTexture()->getData(atcg::GPU).clone();
    _metallic_texture  = material.getMetallicTexture()->getData(atcg::GPU).clone();
    _roughness_texture = material.getRoughnessTexture()->getData(atcg::GPU).clone();

    PBRBSDFData data;

    // Diffuse texture
    {
        cudaTextureDesc desc  = {};
        desc.normalizedCoords = 0;
        desc.addressMode[0]   = cudaAddressModeClamp;
        desc.addressMode[1]   = cudaAddressModeClamp;
        desc.addressMode[2]   = cudaAddressModeClamp;
        desc.readMode         = cudaReadModeElementType;
        desc.filterMode       = cudaFilterModePoint;

        cudaResourceDesc resDesc       = {};
        resDesc.resType                = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr      = _diffuse_texture.data_ptr();
        resDesc.res.linear.sizeInBytes = _diffuse_texture.numel() * _diffuse_texture.element_size();
        resDesc.res.linear.desc.f      = cudaChannelFormatKindUnsigned;
        resDesc.res.linear.desc.x      = 8;
        resDesc.res.linear.desc.y      = 8;
        resDesc.res.linear.desc.z      = 8;
        resDesc.res.linear.desc.w      = 8;

        CUDA_SAFE_CALL(cudaCreateTextureObject(&data.diffuse_texture, &resDesc, &desc, nullptr));
    }

    // Roughness texture
    {
        cudaTextureDesc desc  = {};
        desc.normalizedCoords = 0;
        desc.addressMode[0]   = cudaAddressModeClamp;
        desc.addressMode[1]   = cudaAddressModeClamp;
        desc.addressMode[2]   = cudaAddressModeClamp;
        desc.readMode         = cudaReadModeElementType;
        desc.filterMode       = cudaFilterModeLinear;

        cudaResourceDesc resDesc       = {};
        resDesc.resType                = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr      = _roughness_texture.data_ptr();
        resDesc.res.linear.sizeInBytes = _roughness_texture.numel() * _roughness_texture.element_size();
        resDesc.res.linear.desc.f      = cudaChannelFormatKindFloat;
        resDesc.res.linear.desc.x      = 32;
        resDesc.res.linear.desc.y      = 0;
        resDesc.res.linear.desc.z      = 0;
        resDesc.res.linear.desc.w      = 0;

        CUDA_SAFE_CALL(cudaCreateTextureObject(&data.roughness_texture, &resDesc, &desc, nullptr));
    }

    // Metallic texture
    {
        cudaTextureDesc desc  = {};
        desc.normalizedCoords = 0;
        desc.addressMode[0]   = cudaAddressModeClamp;
        desc.addressMode[1]   = cudaAddressModeClamp;
        desc.addressMode[2]   = cudaAddressModeClamp;
        desc.readMode         = cudaReadModeElementType;
        desc.filterMode       = cudaFilterModeLinear;

        cudaResourceDesc resDesc       = {};
        resDesc.resType                = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr      = _metallic_texture.data_ptr();
        resDesc.res.linear.sizeInBytes = _metallic_texture.numel() * _metallic_texture.element_size();
        resDesc.res.linear.desc.f      = cudaChannelFormatKindFloat;
        resDesc.res.linear.desc.x      = 32;
        resDesc.res.linear.desc.y      = 0;
        resDesc.res.linear.desc.z      = 0;
        resDesc.res.linear.desc.w      = 0;

        CUDA_SAFE_CALL(cudaCreateTextureObject(&data.metallic_texture, &resDesc, &desc, nullptr));
    }

    _bsdf_data_buffer.upload(&data);
}

PBRBSDF::~PBRBSDF() {}

void PBRBSDF::initializeBSDF(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                             const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_bsdf_filename = "C:/Users/Domenic/Documents/Repositories/atcg_framework/build/ptxmodules.dir/"
                                          "Debug/bsdf.ptx";
    auto prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_bsdf"});
    uint32_t idx    = sbt->addCallableEntry(prog_group, _bsdf_data_buffer.get());

    BSDFVPtrTable table;
    table.sampleCallIndex = idx;

    _vptr_table.upload(&table);
}
}    // namespace atcg