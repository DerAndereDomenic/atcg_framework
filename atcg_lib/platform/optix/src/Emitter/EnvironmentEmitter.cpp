#include <Emitter/EnvironmentEmitter.h>

#include <Core/Common.h>

namespace atcg
{

EnvironmentEmitter::EnvironmentEmitter(const Dictionary& dict)
{
    atcg::ref_ptr<atcg::Texture2D> texture = dict.getValue<atcg::ref_ptr<Texture2D>>("environment_texture");

    _flags                   = EmitterFlags::DistantEmitter;
    auto environment_texture = texture->getData(atcg::GPU);

    EnvironmentEmitterData data;

    atcg::convertToTextureObject(environment_texture, _environment_texture, data.environment_texture);

    _environment_emitter_data.upload(&data);
}

EnvironmentEmitter::~EnvironmentEmitter()
{
    EnvironmentEmitterData data;

    _environment_emitter_data.download(&data);

    CUDA_SAFE_CALL(cudaDestroyTextureObject(data.environment_texture));

    CUDA_SAFE_CALL(cudaFreeArray(_environment_texture));
}

void EnvironmentEmitter::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                            const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_emitter_filename = "./bin/EnvironmentEmitter_ptx.ptx";
    auto sample_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__sample_environmentemitter"});
    auto eval_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__eval_environmentemitter"});
    auto evalpdf_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__evalpdf_environmentemitter"});
    uint32_t sample_idx  = sbt->addCallableEntry(sample_prog_group, _environment_emitter_data.get());
    uint32_t eval_idx    = sbt->addCallableEntry(eval_prog_group, _environment_emitter_data.get());
    uint32_t evalpdf_idx = sbt->addCallableEntry(evalpdf_prog_group, _environment_emitter_data.get());

    EmitterVPtrTable table;
    table.flags            = _flags;
    table.sampleCallIndex  = sample_idx;
    table.evalCallIndex    = eval_idx;
    table.evalPdfCallIndex = evalpdf_idx;

    _vptr_table.upload(&table);
}
}    // namespace atcg