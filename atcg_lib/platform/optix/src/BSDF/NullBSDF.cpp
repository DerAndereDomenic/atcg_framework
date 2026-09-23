#include <BSDF/NullBSDF.h>

#include <Core/Common.h>

namespace atcg
{

NullBSDF::NullBSDF(const Dictionary& dict)
{
    _flags = BSDFComponentType::NullTransmission;
}

NullBSDF::~NullBSDF() {}

void NullBSDF::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                  const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_bsdf_filename = "./bin/NullBSDF_ptx.ptx";
    auto sample_prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_nullbsdf"});
    auto eval_prog_group   = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_nullbsdf"});
    uint32_t sample_idx    = sbt->addCallableEntry(sample_prog_group);
    uint32_t eval_idx      = sbt->addCallableEntry(eval_prog_group);

    BSDFVPtrTable table;
    table.sampleCallIndex = sample_idx;
    table.evalCallIndex   = eval_idx;
    table.flags           = _flags;

    _vptr_table.upload(&table);

    markInitialized();
}

void NullBSDF::registerBSDF(BSDFRegistry::Registry* registry)
{
    ATCG_REGISTER_BSDF(registry, "Null", NullBSDF);
}
}    // namespace atcg