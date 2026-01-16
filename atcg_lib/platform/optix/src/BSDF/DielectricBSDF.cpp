#include <BSDF/DielectricBSDF.h>

#include <Renderer/Texture.h>
#include <Renderer/Material.h>
#include <BSDF/BSDFFactory.h>

#include <Core/Common.h>

namespace atcg
{

DielectricBSDF::DielectricBSDF(const Dictionary& dict)
{
    auto material = dict.getValue<atcg::ref_ptr<Material>>("material");

    _diffuse_texture   = material->getDiffuseTexture()->getData(atcg::GPU);
    _roughness_texture = material->getRoughnessTexture()->getData(atcg::GPU);
    _ior_texture       = material->getIorTexture()->getData(atcg::GPU);

    DielectricBSDFData data;

    data.diffuse_texture =
        TextureSampler<glm::vec3>(_diffuse_texture.data_ptr(), material->getDiffuseTexture()->getSpecification());
    data.roughness_texture =
        TextureSampler<float>(_roughness_texture.data_ptr(), material->getRoughnessTexture()->getSpecification());
    data.ior_texture = TextureSampler<float>(_ior_texture.data_ptr(), material->getIorTexture()->getSpecification());

    data.optimizable = _optimizable;

    _flags = BSDFComponentType::IdealReflection | BSDFComponentType::IdealReflection;

    _bsdf_data_buffer.upload(&data);
}

DielectricBSDF::~DielectricBSDF() {}

void PipelineInitializer<DielectricBSDF>::apply(const atcg::ref_ptr<DielectricBSDF>& component) const
{
    const std::string ptx_bsdf_filename = "./bin/DielectricBSDF_ptx.ptx";
    auto sample_prog_group =
        pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_dielectricbsdf"});
    auto eval_prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_dielectricbsdf"});
    auto backward_eval_prog_group =
        pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_backward_dielectricbsdf"});
    auto backward_sample_prog_group =
        pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_backward_dielectricbsdf"});
    auto eval_dual_prog_group =
        pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_forward_dielectricbsdf"});
    auto sample_dual_prog_group =
        pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_forward_dielectricbsdf"});
    uint32_t sample_idx          = sbt->addCallableEntry(sample_prog_group, component->getDataBuffer().get());
    uint32_t eval_idx            = sbt->addCallableEntry(eval_prog_group, component->getDataBuffer().get());
    uint32_t eval_backward_idx   = sbt->addCallableEntry(backward_eval_prog_group, component->getDataBuffer().get());
    uint32_t sample_backward_idx = sbt->addCallableEntry(backward_sample_prog_group, component->getDataBuffer().get());
    uint32_t sample_forward_idx  = sbt->addCallableEntry(sample_dual_prog_group, component->getDataBuffer().get());
    uint32_t eval_forward_idx    = sbt->addCallableEntry(eval_dual_prog_group, component->getDataBuffer().get());

    BSDFVPtrTable table;
    table.sampleCallIndex         = sample_idx;
    table.evalCallIndex           = eval_idx;
    table.evalBackwardCallIndex   = eval_backward_idx;
    table.sampleBackwardCallIndex = sample_backward_idx;
    table.evalForwardCallIndex    = eval_forward_idx;
    table.sampleForwardCallIndex  = sample_forward_idx;
    table.flags                   = component->flags();

    component->getVPtrTableHolder().upload(&table);

    component->markInitialized();
}

std::vector<torch::Tensor> DielectricBSDF::getParameters() const
{
    return {_diffuse_texture, _roughness_texture, _ior_texture};
}

void DielectricBSDF::onImGuiRender()
{
    if(ImGui::Button("Make Optimizable"))
    {
        markOptimizable();
    }

    if(!_diffuse_optimized) return;

    auto normalize = [](torch::Tensor inp) -> torch::Tensor
    {
        auto min = torch::amin(inp);
        auto max = torch::amax(inp);
        auto y   = (inp - min) / (max - min);

        return y;
    };

    auto pos_neg = [normalize](torch::Tensor inp) -> torch::Tensor
    {
        torch::Tensor pos = torch::relu(inp);

        torch::Tensor neg = torch::relu(-inp);

        torch::Tensor y = torch::concat({pos, neg}, /*dim=*/-1);

        return normalize(y);
    };

    _diffuse_optimized->setData(_diffuse_texture);
    _diffuse_grad->setData(normalize(_diffuse_texture.grad()));

    _ior_optimized->setData(_ior_texture);
    _ior_grad->setData(pos_neg(_ior_texture.grad()));

    _roughness_optimized->setData(_roughness_texture);
    _roughness_grad->setData(pos_neg(_roughness_texture.grad()));

    ImGui::Text("Diffuse");
    ImGui::Text("Texture");
    ImGui::Image((ImTextureID)_diffuse_optimized->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
    ImGui::Text("Grad");
    ImGui::Image((ImTextureID)_diffuse_grad->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
    ImGui::Separator();

    ImGui::Text("IoR");
    ImGui::Text("Texture");
    ImGui::Image((ImTextureID)_ior_optimized->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
    ImGui::Text("Grad");
    ImGui::Image((ImTextureID)_ior_grad->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
    ImGui::Separator();

    ImGui::Text("Roughness");
    ImGui::Text("Texture");
    ImGui::Image((ImTextureID)_roughness_optimized->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
    ImGui::Text("Grad");
    ImGui::Image((ImTextureID)_roughness_grad->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
    ImGui::Separator();
}

void DielectricBSDF::markOptimizable()
{
    atcg::TextureSpecification spec_diffuse;
    spec_diffuse.width  = 512;
    spec_diffuse.height = 512;
    spec_diffuse.format = TextureFormat::RGBFLOAT;

    atcg::TextureSpecification spec_float;
    spec_float.width  = 512;
    spec_float.height = 512;
    spec_float.format = TextureFormat::RFLOAT;

    _diffuse_texture   = torch::ones({spec_diffuse.height, spec_diffuse.width, 3},
                                   TensorOptions::floatDeviceOptions().requires_grad(true));
    _ior_texture       = torch::full({spec_float.height, spec_float.height, 1},
                               1.5f,
                               TensorOptions::floatDeviceOptions().requires_grad(true));
    _roughness_texture = torch::zeros({spec_float.height, spec_float.height, 1},
                                      TensorOptions::floatDeviceOptions().requires_grad(true));    // TODO

    _diffuse_texture.mutable_grad()   = torch::zeros_like(_diffuse_texture);
    _ior_texture.mutable_grad()       = torch::zeros_like(_ior_texture);
    _roughness_texture.mutable_grad() = torch::zeros_like(_roughness_texture);

    DielectricBSDFData data;

    data.diffuse_texture   = TextureSampler<glm::vec3>(_diffuse_texture.data_ptr(), spec_diffuse);
    data.ior_texture       = TextureSampler<float>(_ior_texture.data_ptr(), spec_float);
    data.roughness_texture = TextureSampler<float>(_roughness_texture.data_ptr(), spec_float);

    data.diffuse_grad   = TextureSampler<glm::vec3>(_diffuse_texture.grad().data_ptr(), spec_diffuse);
    data.ior_texture    = TextureSampler<float>(_ior_texture.grad().data_ptr(), spec_float);
    data.roughness_grad = TextureSampler<float>(_roughness_texture.grad().data_ptr(), spec_float);

    _bsdf_data_buffer.upload(&data);

    _diffuse_optimized   = atcg::Texture2D::create(spec_diffuse);
    _ior_optimized       = atcg::Texture2D::create(spec_float);
    _roughness_optimized = atcg::Texture2D::create(spec_float);

    spec_float.format = TextureFormat::RGFLOAT;    // For pos/neg visualization
    _diffuse_grad     = atcg::Texture2D::create(spec_diffuse);
    _ior_grad         = atcg::Texture2D::create(spec_float);
    _roughness_grad   = atcg::Texture2D::create(spec_float);

    _optimizable = true;

    DielectricBSDFData bsdf_data;
    _bsdf_data_buffer.download(&bsdf_data);
    bsdf_data.optimizable = _optimizable;
    _bsdf_data_buffer.upload(&bsdf_data);
}

void DielectricBSDF::clampParameters()
{
    _roughness_texture.clamp_(0.0f, 1.0f);
    _ior_texture.clamp_(1.0f, 2.5f);
}

ATCG_REGISTER_BSDF(MaterialType::MATERIAL_TYPE_GLASS, DielectricBSDF);
}    // namespace atcg