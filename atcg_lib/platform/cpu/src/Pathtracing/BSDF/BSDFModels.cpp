#include <Pathtracing/BSDFModels.h>
#include <DataStructure/TorchUtils.h>

namespace atcg
{
PBRBSDF::PBRBSDF(const Material& material)
{
    PBRBSDFData data;
    data.diffuse_texture   = material.getDiffuseTexture()->getData(atcg::CPU);
    data.roughness_texture = material.getRoughnessTexture()->getData(atcg::CPU);
    data.metallic_texture  = material.getMetallicTexture()->getData(atcg::CPU);

    _bsdf_data_buffer = atcg::make_ref<PBRBSDFData>(data);
}

void PBRBSDF::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                 const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    _vptr_table = atcg::make_ref<BSDFVPtrTable>();

    auto ptx_file     = "C:/Users/Domenic/Documents/Repositories/atcg_framework/bin/Debug/BSDFModels.dll";
    auto sample_group = pipeline->addCallableShader({ptx_file, "__direct_callable__sample_pbrbsdf"});
    auto eval_group   = pipeline->addCallableShader({ptx_file, "__direct_callable__eval_pbrbsdf"});

    auto sample_idx = sbt->addCallableEntry(sample_group, _bsdf_data_buffer.get());
    auto eval_idx   = sbt->addCallableEntry(eval_group, _bsdf_data_buffer.get());

    _vptr_table->sampleCallIndex = sample_idx;
    _vptr_table->evalCallIndex   = eval_idx;
    _vptr_table->flags           = _flags;
}

PBRBSDF::~PBRBSDF() {}

RefractiveBSDF::RefractiveBSDF(const Material& material)
{
    RefractiveBSDFData data;
    data.diffuse_texture = material.getDiffuseTexture()->getData(atcg::CPU);
    data.ior             = material.ior;

    _bsdf_data_buffer = atcg::make_ref<RefractiveBSDFData>(data);
}

RefractiveBSDF::~RefractiveBSDF() {}

void RefractiveBSDF::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                        const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    _vptr_table = atcg::make_ref<BSDFVPtrTable>();

    auto ptx_file     = "C:/Users/Domenic/Documents/Repositories/atcg_framework/bin/Debug/BSDFModels.dll";
    auto sample_group = pipeline->addCallableShader({ptx_file, "__direct_callable__sample_refractivebsdf"});
    auto eval_group   = pipeline->addCallableShader({ptx_file, "__direct_callable__eval_refractivebsdf"});

    auto sample_idx = sbt->addCallableEntry(sample_group, _bsdf_data_buffer.get());
    auto eval_idx   = sbt->addCallableEntry(eval_group, _bsdf_data_buffer.get());

    _vptr_table->sampleCallIndex = sample_idx;
    _vptr_table->evalCallIndex   = eval_idx;
    _vptr_table->flags           = _flags;
}
}    // namespace atcg