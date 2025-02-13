#include <Emitter/PointEmitter.h>

namespace atcg
{
PointEmitter::PointEmitter(const glm::vec3& position, const PointLightComponent& point_light)
{
    _flags = EmitterFlags::InfinitesimalSize;
    PointEmitterData data;
    data.position  = position;
    data.intensity = point_light.intensity;
    data.color     = point_light.color;

    _point_emitter_data.upload(&data);
}

PointEmitter::~PointEmitter() {}

void PointEmitter::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                      const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_emitter_filename = "./bin/PointEmitter_ptx.ptx";
    auto sample_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__sample_pointemitter"});
    auto eval_prog_group = pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__eval_pointemitter"});
    auto evalpdf_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__evalpdf_pointemitter"});
    uint32_t sample_idx  = sbt->addCallableEntry(sample_prog_group, _point_emitter_data.get());
    uint32_t eval_idx    = sbt->addCallableEntry(eval_prog_group, _point_emitter_data.get());
    uint32_t evalpdf_idx = sbt->addCallableEntry(evalpdf_prog_group, _point_emitter_data.get());

    EmitterVPtrTable table;
    table.flags            = _flags;
    table.sampleCallIndex  = sample_idx;
    table.evalCallIndex    = eval_idx;
    table.evalPdfCallIndex = evalpdf_idx;

    _vptr_table.upload(&table);
}
}    // namespace atcg