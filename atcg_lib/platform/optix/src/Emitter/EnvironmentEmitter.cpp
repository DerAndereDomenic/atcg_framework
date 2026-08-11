#include <Emitter/EnvironmentEmitter.h>

#include <Core/Optix.h>

namespace atcg
{

EnvironmentEmitter::EnvironmentEmitter(const Dictionary& dict)
{
    atcg::ref_ptr<atcg::Texture2D> texture = dict.getValue<atcg::ref_ptr<Texture2D>>("environment_texture");
    atcg::BoundingBox scene_aabb           = dict.getValueOr<atcg::BoundingBox>("scene_aabb", atcg::BoundingBox());

    _flags               = EmitterFlags::DistantEmitter;
    _environment_texture = std::dynamic_pointer_cast<Texture2D>(texture->clone());

    torch::Tensor env_map = texture->getData(atcg::GPU);

    if(env_map.dim() == 3 && env_map.size(2) == 4)
    {
        env_map = env_map.slice(2, 0, 3);
    }

    if(env_map.dtype() != torch::kFloat32)
    {
        env_map = env_map.to(torch::kFloat32) / 255.0f;
    }

    torch::Tensor weights = torch::mean(env_map, -1).flip({0});
    torch::Tensor theta   = glm::pi<float>() * (1.0f - (torch::arange(env_map.size(0), atcg::GPU)) / env_map.size(0));
    weights               = weights * torch::sin(theta).unsqueeze(-1);

    _row_pdf = torch::sum(weights, 1);
    _row_cdf = torch::cumsum(_row_pdf, 0);
    _row_pdf = _row_pdf / _row_cdf[-1];
    _row_cdf = _row_cdf / _row_cdf[-1];

    _col_pdfs = weights / (torch::sum(weights, 1, true) + 1e-5f);
    _col_cdfs = torch::cumsum(_col_pdfs, 1);

    EnvironmentEmitterData data;

    data.environment_texture.texture_data.texture = _environment_texture->getTextureObject();
    data.environment_texture.spec                 = _environment_texture->getSpecification();
    data.col_pdfs                                 = _col_pdfs.data_ptr<float>();
    data.col_cdfs                                 = _col_cdfs.data_ptr<float>();
    data.row_pdf                                  = _row_pdf.data_ptr<float>();
    data.row_cdf                                  = _row_cdf.data_ptr<float>();
    data.width                                    = env_map.size(1);
    data.height                                   = env_map.size(0);
    data.bounding_box                             = scene_aabb;

    _environment_emitter_data.upload(&data);
}

EnvironmentEmitter::~EnvironmentEmitter()
{
    _environment_texture->unmapDevicePointers();
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
    auto sample_photon_prog_group =
        pipeline->addCallableShader({ptx_emitter_filename, "__direct_callable__samplephoton_environmentemitter"});
    uint32_t sample_idx        = sbt->addCallableEntry(sample_prog_group, _environment_emitter_data.get());
    uint32_t eval_idx          = sbt->addCallableEntry(eval_prog_group, _environment_emitter_data.get());
    uint32_t evalpdf_idx       = sbt->addCallableEntry(evalpdf_prog_group, _environment_emitter_data.get());
    uint32_t sample_photon_idx = sbt->addCallableEntry(sample_photon_prog_group, _environment_emitter_data.get());

    EmitterVPtrTable table;
    table.flags                 = _flags;
    table.sampleCallIndex       = sample_idx;
    table.evalCallIndex         = eval_idx;
    table.evalPdfCallIndex      = evalpdf_idx;
    table.samplePhotonCallIndex = sample_photon_idx;

    _vptr_table.upload(&table);

    markInitialized();
}
}    // namespace atcg