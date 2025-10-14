#include <Emitter/EnvironmentEmitter.h>

#include <Core/Common.h>

namespace atcg
{

EnvironmentEmitter::EnvironmentEmitter(const Dictionary& dict)
{
    atcg::ref_ptr<atcg::Texture2D> texture = dict.getValue<atcg::ref_ptr<Texture2D>>("environment_texture");

    _flags               = EmitterFlags::DistantEmitter;
    _environment_texture = std::dynamic_pointer_cast<Texture2D>(texture->clone());

    EnvironmentEmitterData data;

    data.environment_texture.texture_data.texture = _environment_texture->getTextureObject();
    data.environment_texture.spec                 = _environment_texture->getSpecification();

    _environment_emitter_data.upload(&data);
}

EnvironmentEmitter::~EnvironmentEmitter()
{
    _environment_texture->unmapDevicePointers();
}

void PipelineInitializer<EnvironmentEmitter>::apply(const atcg::ref_ptr<EnvironmentEmitter>& component) const
{
    const std::string ptx_emitter_filename = "./bin/EnvironmentEmitter_ptx.ptx";
    auto sample_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__sample_environmentemitter"});
    auto eval_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__eval_environmentemitter"});
    auto evalpdf_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__evalpdf_environmentemitter"});
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