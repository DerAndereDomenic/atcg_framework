#include <Medium/RayleighPhaseFunction.h>

namespace atcg
{
RayleighPhaseFunction::RayleighPhaseFunction(const Dictionary& dict) : PhaseFunction(dict) {}

RayleighPhaseFunction::~RayleighPhaseFunction() {}

void RayleighPhaseFunction::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                               const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_bsdf_filename = "./bin/RayleighPhaseFunction_ptx.ptx";
    auto sample_prog_group =
        pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_rayleighphase"});
    auto eval_prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_rayleighphase"});
    uint32_t sample_idx  = sbt->addCallableEntry(sample_prog_group, nullptr);
    uint32_t eval_idx    = sbt->addCallableEntry(eval_prog_group, nullptr);

    PhaseFunctionVPtrTable table;
    table.sampleCallIndex = sample_idx;
    table.evalCallIndex   = eval_idx;

    _vptr_table.upload(&table);

    markInitialized();
}
}