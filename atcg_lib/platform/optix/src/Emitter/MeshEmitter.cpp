#include <Emitter/MeshEmitter.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <Shape/MeshShape.h>
#include <Core/Common.h>

namespace atcg
{
MeshEmitter::MeshEmitter(const Dictionary& dict)
{
    auto shape            = dict.getValue<atcg::ref_ptr<MeshShape>>("shape");
    auto texture_emissive = dict.getValue<atcg::ref_ptr<Texture2D>>("texture_emissive");
    auto emission_scaling = dict.getValue<float>("emission_scaling");
    glm::mat4 transform   = dict.getValue<glm::mat4>("transform");

    MeshEmitterData data;

    _emissive_texture                          = std::dynamic_pointer_cast<Texture2D>(texture_emissive->clone());
    data.emissive_texture.texture_data.texture = _emissive_texture->getTextureObject();
    data.emissive_texture.spec                 = _emissive_texture->getSpecification();

    data.emitter_scaling = emission_scaling;

    _sampler     = shape->createSampler(transform);
    data.sampler = _sampler->getVPtrTable();

    _mesh_emitter_data.upload(&data);
}

MeshEmitter::~MeshEmitter()
{
    _emissive_texture->unmapDevicePointers();
}

void MeshEmitter::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                     const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    _sampler->ensureInitialized(pipeline, sbt);

    const std::string ptx_emitter_filename = "./bin/MeshEmitter_ptx.ptx";
    auto sample_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__sample_meshemitter"});
    auto eval_prog_group = pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__eval_meshemitter"});
    auto evalpdf_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__evalpdf_meshemitter"});
    auto samplephoton_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__samplephoton_meshemitter"});

    uint32_t sample_idx       = sbt->addCallableEntry(sample_prog_group, _mesh_emitter_data.get());
    uint32_t eval_idx         = sbt->addCallableEntry(eval_prog_group, _mesh_emitter_data.get());
    uint32_t eval_pdf_idx     = sbt->addCallableEntry(evalpdf_prog_group, _mesh_emitter_data.get());
    uint32_t samplephoton_idx = sbt->addCallableEntry(samplephoton_prog_group, _mesh_emitter_data.get());

    EmitterVPtrTable table;
    table.flags                 = _flags;
    table.sampleCallIndex       = sample_idx;
    table.evalCallIndex         = eval_idx;
    table.evalPdfCallIndex      = eval_pdf_idx;
    table.samplePhotonCallIndex = samplephoton_idx;

    _vptr_table.upload(&table);

    markInitialized();
}
}    // namespace atcg