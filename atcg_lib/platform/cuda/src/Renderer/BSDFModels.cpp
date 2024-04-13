#include <Renderer/BSDFModels.h>
#include <Core/CUDA.h>

namespace atcg
{
PBRBSDF::PBRBSDF(const Material& material)
{
    auto diffuse_texture   = material.getDiffuseTexture()->getData(atcg::GPU);
    auto metallic_texture  = material.getMetallicTexture()->getData(atcg::GPU);
    auto roughness_texture = material.getRoughnessTexture()->getData(atcg::GPU);

    PBRBSDFData data;

    // Diffuse texture
    {
        cudaChannelFormatDesc format = {};
        format.f                     = cudaChannelFormatKindUnsigned;
        format.x                     = 8;
        format.y                     = 8;
        format.z                     = 8;
        format.w                     = 8;

        cudaExtent extent = {};
        extent.width      = diffuse_texture.size(1);
        extent.height     = diffuse_texture.size(0);
        extent.depth      = 0;

        CUDA_SAFE_CALL(cudaMalloc3DArray(&_diffuse_texture, &format, extent));

        CUDA_SAFE_CALL(
            cudaMemcpy2DToArray(_diffuse_texture,
                                0,
                                0,
                                diffuse_texture.contiguous().data_ptr(),
                                diffuse_texture.size(1) * diffuse_texture.size(2) * diffuse_texture.element_size(),
                                diffuse_texture.size(1) * diffuse_texture.size(2) * diffuse_texture.element_size(),
                                diffuse_texture.size(0),
                                cudaMemcpyDeviceToDevice));

        cudaTextureDesc desc  = {};
        desc.normalizedCoords = 1;
        desc.addressMode[0]   = cudaAddressModeClamp;
        desc.addressMode[1]   = cudaAddressModeClamp;
        desc.addressMode[2]   = cudaAddressModeClamp;
        desc.readMode         = cudaReadModeNormalizedFloat;
        desc.filterMode       = cudaFilterModeLinear;

        cudaResourceDesc resDesc = {};
        resDesc.resType          = cudaResourceTypeArray;
        resDesc.res.array.array  = _diffuse_texture;

        CUDA_SAFE_CALL(cudaCreateTextureObject(&data.diffuse_texture, &resDesc, &desc, nullptr));
    }

    // Roughness texture
    {
        cudaChannelFormatDesc format = {};
        format.f                     = cudaChannelFormatKindFloat;
        format.x                     = 32;
        format.y                     = 0;
        format.z                     = 0;
        format.w                     = 0;

        cudaExtent extent = {};
        extent.width      = roughness_texture.size(1);
        extent.height     = roughness_texture.size(0);
        extent.depth      = 0;

        CUDA_SAFE_CALL(cudaMalloc3DArray(&_roughness_texture, &format, extent));

        CUDA_SAFE_CALL(cudaMemcpy2DToArray(
            _roughness_texture,
            0,
            0,
            roughness_texture.contiguous().data_ptr(),
            roughness_texture.size(1) * roughness_texture.size(2) * roughness_texture.element_size(),
            roughness_texture.size(1) * roughness_texture.size(2) * roughness_texture.element_size(),
            roughness_texture.size(0),
            cudaMemcpyDeviceToDevice));

        cudaTextureDesc desc  = {};
        desc.normalizedCoords = 1;
        desc.addressMode[0]   = cudaAddressModeClamp;
        desc.addressMode[1]   = cudaAddressModeClamp;
        desc.addressMode[2]   = cudaAddressModeClamp;
        desc.readMode         = cudaReadModeElementType;
        desc.filterMode       = cudaFilterModeLinear;

        cudaResourceDesc resDesc = {};
        resDesc.resType          = cudaResourceTypeArray;
        resDesc.res.array.array  = _roughness_texture;

        CUDA_SAFE_CALL(cudaCreateTextureObject(&data.roughness_texture, &resDesc, &desc, nullptr));
    }

    // Metallic texture
    {
        cudaChannelFormatDesc format = {};
        format.f                     = cudaChannelFormatKindFloat;
        format.x                     = 32;
        format.y                     = 0;
        format.z                     = 0;
        format.w                     = 0;

        cudaExtent extent = {};
        extent.width      = metallic_texture.size(1);
        extent.height     = metallic_texture.size(0);
        extent.depth      = 0;

        CUDA_SAFE_CALL(cudaMalloc3DArray(&_metallic_texture, &format, extent));

        CUDA_SAFE_CALL(
            cudaMemcpy2DToArray(_metallic_texture,
                                0,
                                0,
                                metallic_texture.contiguous().data_ptr(),
                                metallic_texture.size(1) * metallic_texture.size(2) * metallic_texture.element_size(),
                                metallic_texture.size(1) * metallic_texture.size(2) * metallic_texture.element_size(),
                                metallic_texture.size(0),
                                cudaMemcpyDeviceToDevice));

        cudaTextureDesc desc  = {};
        desc.normalizedCoords = 1;
        desc.addressMode[0]   = cudaAddressModeClamp;
        desc.addressMode[1]   = cudaAddressModeClamp;
        desc.addressMode[2]   = cudaAddressModeClamp;
        desc.readMode         = cudaReadModeElementType;
        desc.filterMode       = cudaFilterModeLinear;

        cudaResourceDesc resDesc = {};
        resDesc.resType          = cudaResourceTypeArray;
        resDesc.res.array.array  = _metallic_texture;

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