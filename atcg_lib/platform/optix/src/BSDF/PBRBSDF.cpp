#include <BSDF/PBRBSDF.h>

#include <Core/Common.h>
#include <BSDF/BSDFFactory.h>
#include <Renderer/Texture.h>

namespace atcg
{

PBRBSDF::PBRBSDF(const Dictionary& dict)
{
    atcg::ref_ptr<Material> material = dict.getValue<atcg::ref_ptr<Material>>("material");

    _diffuse_texture   = material->getDiffuseTexture()->getData(atcg::GPU);
    _metallic_texture  = material->getMetallicTexture()->getData(atcg::GPU);
    _roughness_texture = material->getRoughnessTexture()->getData(atcg::GPU);

    PBRBSDFData data;

    data.diffuse_texture =
        TextureSampler<glm::vec3>(_diffuse_texture.data_ptr(), material->getDiffuseTexture()->getSpecification());
    data.metallic_texture =
        TextureSampler<float>(_metallic_texture.data_ptr(), material->getMetallicTexture()->getSpecification());
    data.roughness_texture =
        TextureSampler<float>(_roughness_texture.data_ptr(), material->getRoughnessTexture()->getSpecification());

    data.optimizable = _optimizable;

    _flags = BSDFComponentType::GlossyReflection | BSDFComponentType::DiffuseReflection;

    _bsdf_data_buffer.upload(&data);
}

PBRBSDF::~PBRBSDF() {}

void PipelineInitializer<PBRBSDF>::apply(const atcg::ref_ptr<PBRBSDF>& component) const
{
    const std::string ptx_bsdf_filename = "./bin/PBRBSDF_ptx.ptx";

    auto sample_prog_group   = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_pbrbsdf"});
    auto eval_prog_group     = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_pbrbsdf"});
    auto backward_prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__grad_pbrbsdf"});
    uint32_t sample_idx      = sbt->addCallableEntry(sample_prog_group, component->getDataBuffer().get());
    uint32_t eval_idx        = sbt->addCallableEntry(eval_prog_group, component->getDataBuffer().get());
    uint32_t backward_idx    = sbt->addCallableEntry(backward_prog_group, component->getDataBuffer().get());

    BSDFVPtrTable table;
    table.sampleCallIndex   = sample_idx;
    table.evalCallIndex     = eval_idx;
    table.evalBackwardIndex = backward_idx;
    table.flags             = component->flags();

    component->getVPtrTableHolder().upload(&table);

    component->markInitialized();
}

std::vector<torch::Tensor> PBRBSDF::getParameters() const
{
    return {_diffuse_texture, _metallic_texture, _roughness_texture};
}

void PBRBSDF::onImGuiRender()
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

    _metallic_optimized->setData(_metallic_texture);
    _metallic_grad->setData(pos_neg(_metallic_texture.grad()));

    _roughness_optimized->setData(_roughness_texture);
    _roughness_grad->setData(pos_neg(_roughness_texture.grad()));

    ImGui::Text("Diffuse");
    ImGui::Text("Texture");
    ImGui::Image((ImTextureID)_diffuse_optimized->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
    ImGui::Text("Grad");
    ImGui::Image((ImTextureID)_diffuse_grad->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
    ImGui::Separator();

    ImGui::Text("Metallic");
    ImGui::Text("Texture");
    ImGui::Image((ImTextureID)_metallic_optimized->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
    ImGui::Text("Grad");
    ImGui::Image((ImTextureID)_metallic_grad->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
    ImGui::Separator();

    ImGui::Text("Roughness");
    ImGui::Text("Texture");
    ImGui::Image((ImTextureID)_roughness_optimized->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
    ImGui::Text("Grad");
    ImGui::Image((ImTextureID)_roughness_grad->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
    ImGui::Separator();
}

void PBRBSDF::markOptimizable()
{
    atcg::TextureSpecification spec_diffuse;
    spec_diffuse.width  = 512;
    spec_diffuse.height = 512;
    spec_diffuse.format = TextureFormat::RGBFLOAT;

    atcg::TextureSpecification spec_float;
    spec_float.width  = 512;
    spec_float.height = 512;
    spec_float.format = TextureFormat::RFLOAT;

    _diffuse_texture   = torch::zeros({spec_diffuse.height, spec_diffuse.width, 3},
                                    TensorOptions::floatDeviceOptions().requires_grad(true));
    _metallic_texture  = torch::zeros({spec_float.height, spec_float.height, 1},
                                     TensorOptions::floatDeviceOptions().requires_grad(true));
    _roughness_texture = torch::ones({spec_float.height, spec_float.height, 1},
                                     TensorOptions::floatDeviceOptions().requires_grad(true));    // TODO

    _diffuse_texture.mutable_grad()   = torch::zeros_like(_diffuse_texture);
    _metallic_texture.mutable_grad()  = torch::zeros_like(_metallic_texture);
    _roughness_texture.mutable_grad() = torch::zeros_like(_roughness_texture);

    PBRBSDFData data;

    data.diffuse_texture   = TextureSampler<glm::vec3>(_diffuse_texture.data_ptr(), spec_diffuse);
    data.metallic_texture  = TextureSampler<float>(_metallic_texture.data_ptr(), spec_float);
    data.roughness_texture = TextureSampler<float>(_roughness_texture.data_ptr(), spec_float);

    data.diffuse_grad   = TextureSampler<glm::vec3>(_diffuse_texture.grad().data_ptr(), spec_diffuse);
    data.metallic_grad  = TextureSampler<float>(_metallic_texture.grad().data_ptr(), spec_float);
    data.roughness_grad = TextureSampler<float>(_roughness_texture.grad().data_ptr(), spec_float);

    _bsdf_data_buffer.upload(&data);

    _diffuse_optimized   = atcg::Texture2D::create(spec_diffuse);
    _metallic_optimized  = atcg::Texture2D::create(spec_float);
    _roughness_optimized = atcg::Texture2D::create(spec_float);

    spec_float.format = TextureFormat::RGFLOAT;    // For pos/neg visualization
    _diffuse_grad     = atcg::Texture2D::create(spec_diffuse);
    _metallic_grad    = atcg::Texture2D::create(spec_float);
    _roughness_grad   = atcg::Texture2D::create(spec_float);

    _optimizable = true;

    PBRBSDFData bsdf_data;
    _bsdf_data_buffer.download(&bsdf_data);
    bsdf_data.optimizable = _optimizable;
    _bsdf_data_buffer.upload(&bsdf_data);
}

ATCG_REGISTER_BSDF(MaterialType::MATERIAL_TYPE_OPAQUE, PBRBSDF);
}    // namespace atcg