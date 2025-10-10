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

void PipelineInitializer<HenyeyGreensteinPhaseFunction>::apply(
    const atcg::ref_ptr<HenyeyGreensteinPhaseFunction>& component) const
{
    const std::string ptx_bsdf_filename = "./bin/HenyeyGreensteinPhaseFunction_ptx.ptx";
    auto sample_prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_hgphase"});
    auto eval_prog_group   = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_hgphase"});
    uint32_t sample_idx    = sbt->addCallableEntry(sample_prog_group, component->getDataBuffer().get());
    uint32_t eval_idx      = sbt->addCallableEntry(eval_prog_group, component->getDataBuffer().get());

    PhaseFunctionVPtrTable table;
    table.sampleCallIndex = sample_idx;
    table.evalCallIndex   = eval_idx;

    component->getVPtrTableHolder().upload(&table);

    component->markInitialized();
}
}    // namespace atcg