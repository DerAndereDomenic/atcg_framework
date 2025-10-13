#include <Emitter/PointEmitter.h>

namespace atcg
{
PointEmitter::PointEmitter(const atcg::Dictionary& dict)
{
    _flags = EmitterFlags::InfinitesimalSize;
    PointEmitterData data;
    data.position  = dict.getValue<glm::vec3>("position");
    data.intensity = dict.getValue<float>("intensity");
    data.color     = dict.getValue<glm::vec3>("color");

    _point_emitter_data.upload(&data);
}

PointEmitter::~PointEmitter() {}

void PipelineInitializer<PointEmitter>::apply(const atcg::ref_ptr<PointEmitter>& component) const
{
    const std::string ptx_emitter_filename = "./bin/PointEmitter_ptx.ptx";
    auto sample_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__sample_pointemitter"});
    auto eval_prog_group = pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__eval_pointemitter"});
    auto evalpdf_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__evalpdf_pointemitter"});
    uint32_t sample_idx  = sbt->addCallableEntry(sample_prog_group, component->getDataBuffer().get());
    uint32_t eval_idx    = sbt->addCallableEntry(eval_prog_group, component->getDataBuffer().get());
    uint32_t evalpdf_idx = sbt->addCallableEntry(evalpdf_prog_group, component->getDataBuffer().get());

    EmitterVPtrTable table;
    table.flags            = component->flags();
    table.sampleCallIndex  = sample_idx;
    table.evalCallIndex    = eval_idx;
    table.evalPdfCallIndex = evalpdf_idx;

    component->getVPtrTableHolder().upload(&table);

    component->markInitialized();
}
}    // namespace atcg