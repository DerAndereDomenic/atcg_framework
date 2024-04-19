#include <Renderer/BSDFModels.h>
#include <Core/CUDA.h>

#include <Renderer/Common.h>

namespace atcg
{

PBRBSDF::PBRBSDF(const Material& material)
{
    auto diffuse_texture   = material.getDiffuseTexture()->getData(atcg::GPU);
    auto metallic_texture  = material.getMetallicTexture()->getData(atcg::GPU);
    auto roughness_texture = material.getRoughnessTexture()->getData(atcg::GPU);

    PBRBSDFData data;

    ::detail::convertToTextureObject(diffuse_texture, _diffuse_texture, data.diffuse_texture);
    ::detail::convertToTextureObject(metallic_texture, _metallic_texture, data.metallic_texture);
    ::detail::convertToTextureObject(roughness_texture, _roughness_texture, data.roughness_texture);

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

void PBRBSDF::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                 const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_bsdf_filename = "./build/ptxmodules.dir/Debug/BSDFKernels.ptx";
    auto sample_prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_pbrbsdf"});
    auto eval_prog_group   = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_pbrbsdf"});
    uint32_t sample_idx    = sbt->addCallableEntry(sample_prog_group, _bsdf_data_buffer.get());
    uint32_t eval_idx      = sbt->addCallableEntry(eval_prog_group, _bsdf_data_buffer.get());

    BSDFVPtrTable table;
    table.sampleCallIndex = sample_idx;
    table.evalCallIndex   = eval_idx;

    _vptr_table.upload(&table);
}

RefractiveBSDF::RefractiveBSDF(const Material& material)
{
    auto diffuse_texture = material.getDiffuseTexture()->getData(atcg::GPU);

    RefractiveBSDFData data;

    ::detail::convertToTextureObject(diffuse_texture, _diffuse_texture, data.diffuse_texture);
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

void RefractiveBSDF::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                        const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_bsdf_filename = "./build/ptxmodules.dir/Debug/BSDFKernels.ptx";
    auto sample_prog_group =
        pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_refractivebsdf"});
    auto eval_prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_refractivebsdf"});
    uint32_t sample_idx  = sbt->addCallableEntry(sample_prog_group, _bsdf_data_buffer.get());
    uint32_t eval_idx    = sbt->addCallableEntry(eval_prog_group, _bsdf_data_buffer.get());

    BSDFVPtrTable table;
    table.sampleCallIndex = sample_idx;
    table.evalCallIndex   = eval_idx;

    _vptr_table.upload(&table);
}
}    // namespace atcg