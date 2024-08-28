#include <Pathtracing/BSDFModels.h>
#include <DataStructure/TorchUtils.h>

#include <Pathtracing/Common.h>

namespace atcg
{
PBRBSDF::PBRBSDF(const Material& material)
{
    auto diffuse_texture   = material.getDiffuseTexture()->getData(atcg::GPU);
    auto metallic_texture  = material.getMetallicTexture()->getData(atcg::GPU);
    auto roughness_texture = material.getRoughnessTexture()->getData(atcg::GPU);

    PBRBSDFData data;

    // TODO: Convert to texture method
    ::detail::convertToTextureObject(diffuse_texture, _diffuse_texture, data.diffuse_texture);
    ::detail::convertToTextureObject(metallic_texture, _metallic_texture, data.metallic_texture);
    ::detail::convertToTextureObject(roughness_texture, _roughness_texture, data.roughness_texture);

    _flags = BSDFComponentType::GlossyReflection | BSDFComponentType::DiffuseReflection;

    _bsdf_data_buffer.upload(&data);
}

// TODO: Destructor

void PBRBSDF::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                 const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    auto ptx_file     = "./bin/BSDFModels.ptx";
    auto sample_group = pipeline->addCallableShader({ptx_file, "__direct_callable__sample_pbrbsdf"});
    auto eval_group   = pipeline->addCallableShader({ptx_file, "__direct_callable__eval_pbrbsdf"});

    auto sample_idx = sbt->addCallableEntry(sample_group, _bsdf_data_buffer.get());
    auto eval_idx   = sbt->addCallableEntry(eval_group, _bsdf_data_buffer.get());

    BSDFVPtrTable table;
    table.sampleCallIndex = sample_idx;
    table.evalCallIndex   = eval_idx;
    table.flags           = _flags;

    _vptr_table.upload(&table);
}

PBRBSDF::~PBRBSDF() {}

RefractiveBSDF::RefractiveBSDF(const Material& material)
{
    auto diffuse_texture = material.getDiffuseTexture()->getData(atcg::GPU);

    RefractiveBSDFData data;

    // TODO: Convert texture
    ::detail::convertToTextureObject(diffuse_texture, _diffuse_texture, data.diffuse_texture);
    data.ior = material.ior;

    _flags = BSDFComponentType::IdealReflection | BSDFComponentType::IdealTransmission;

    _bsdf_data_buffer.upload(&data);
}

// TODO:
RefractiveBSDF::~RefractiveBSDF() {}

void RefractiveBSDF::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                        const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    auto ptx_file     = "./bin/BSDFModels.ptx";
    auto sample_group = pipeline->addCallableShader({ptx_file, "__direct_callable__sample_refractivebsdf"});
    auto eval_group   = pipeline->addCallableShader({ptx_file, "__direct_callable__eval_refractivebsdf"});

    auto sample_idx = sbt->addCallableEntry(sample_group, _bsdf_data_buffer.get());
    auto eval_idx   = sbt->addCallableEntry(eval_group, _bsdf_data_buffer.get());

    BSDFVPtrTable table;
    table.sampleCallIndex = sample_idx;
    table.evalCallIndex   = eval_idx;
    table.flags           = _flags;

    _vptr_table.upload(&table);
}
}    // namespace atcg