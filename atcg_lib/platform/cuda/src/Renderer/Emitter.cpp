#include <Renderer/Emitter.h>
#include <Renderer/Common.h>

namespace atcg
{
MeshEmitter::MeshEmitter(const atcg::ref_ptr<Graph>& graph, const Material& material)
{
    auto emissive_texture = material.getEmissiveTexture()->getData(atcg::GPU);

    MeshEmitterData data;

    ::detail::convertToTextureObject(emissive_texture, _emissive_texture, data.emissive_texture);

    data.emitter_scaling = material.emission_scale;

    _positions = graph->getPositions(atcg::GPU).clone();
    _normals   = graph->getNormals(atcg::GPU).clone();
    _uvs       = graph->getUVs(atcg::GPU).clone();
    _faces     = graph->getFaces(atcg::GPU).clone();

    data.positions = (glm::vec3*)_positions.data_ptr();
    data.normals   = (glm::vec3*)_positions.data_ptr();
    data.uvs       = (glm::vec3*)_positions.data_ptr();
    data.faces     = (glm::u32vec3*)_positions.data_ptr();

    _mesh_emitter_data.upload(&data);
}

MeshEmitter::~MeshEmitter()
{
    MeshEmitterData data;

    _mesh_emitter_data.download(&data);

    CUDA_SAFE_CALL(cudaDestroyTextureObject(data.emissive_texture));

    CUDA_SAFE_CALL(cudaFreeArray(_emissive_texture));
}

void MeshEmitter::initializeEmitter(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_emitter_filename = "./build/ptxmodules.dir/Debug/emitter.ptx";
    auto sample_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__sample_meshemitter"});
    auto eval_prog_group = pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__eval_meshemitter"});
    uint32_t sample_idx  = sbt->addCallableEntry(sample_prog_group, _mesh_emitter_data.get());
    uint32_t eval_idx    = sbt->addCallableEntry(eval_prog_group, _mesh_emitter_data.get());

    EmitterVPtrTable table;
    table.sampleCallIndex = sample_idx;
    table.evalCallIndex   = eval_idx;

    _vptr_table.upload(&table);
}

EnvironmentEmitter::EnvironmentEmitter(const atcg::ref_ptr<Texture2D>& texture)
{
    auto environment_texture = texture->getData(atcg::GPU);

    EnvironmentEmitterData data;

    ::detail::convertToTextureObject(environment_texture, _environment_texture, data.environment_texture);

    _environment_emitter_data.upload(&data);
}

EnvironmentEmitter::~EnvironmentEmitter()
{
    EnvironmentEmitterData data;

    _environment_emitter_data.download(&data);

    CUDA_SAFE_CALL(cudaDestroyTextureObject(data.environment_texture));

    CUDA_SAFE_CALL(cudaFreeArray(_environment_texture));
}

void EnvironmentEmitter::initializeEmitter(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                           const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_emitter_filename = "./build/ptxmodules.dir/Debug/emitter.ptx";
    auto sample_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__sample_environmentemitter"});
    auto eval_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__eval_environmentemitter"});
    uint32_t sample_idx = sbt->addCallableEntry(sample_prog_group, _environment_emitter_data.get());
    uint32_t eval_idx   = sbt->addCallableEntry(eval_prog_group, _environment_emitter_data.get());

    EmitterVPtrTable table;
    table.sampleCallIndex = sample_idx;
    table.evalCallIndex   = eval_idx;

    _vptr_table.upload(&table);
}
}    // namespace atcg