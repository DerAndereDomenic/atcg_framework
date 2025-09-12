#include <Medium/HenyeyGreensteinPhaseFunction.h>

namespace atcg
{
HenyeyGreensteinPhaseFunction::HenyeyGreensteinPhaseFunction(const atcg::Dictionary& dict) : PhaseFunction(dict)
{
    float g = dict.getValueOr<float>("g", 0.0f);

    HenyeyGreensteinPhaseFunctionData data;
    data.g = g;

    _data_buffer.upload(&data);
}

HenyeyGreensteinPhaseFunction::~HenyeyGreensteinPhaseFunction() {}

void HenyeyGreensteinPhaseFunction::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                                       const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_bsdf_filename = "./bin/HenyeyGreensteinPhaseFunction_ptx.ptx";
    auto sample_prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_hgphase"});
    auto eval_prog_group   = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_hgphase"});
    uint32_t sample_idx    = sbt->addCallableEntry(sample_prog_group, _data_buffer.get());
    uint32_t eval_idx      = sbt->addCallableEntry(eval_prog_group, _data_buffer.get());

    PhaseFunctionVPtrTable table;
    table.sampleCallIndex = sample_idx;
    table.evalCallIndex   = eval_idx;

    _vptr_table.upload(&table);
}
}    // namespace atcg