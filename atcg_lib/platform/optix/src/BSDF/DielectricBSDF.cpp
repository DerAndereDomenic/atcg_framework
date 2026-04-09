#include <BSDF/DielectricBSDF.h>

#include <Renderer/Texture.h>
#include <Renderer/Material.h>
#include <BSDF/BSDFFactory.h>

#include <Core/Common.h>

namespace atcg
{

DielectricBSDF::DielectricBSDF(const Dictionary& dict)
{
    auto material = std::dynamic_pointer_cast<atcg::DielectricMaterial>(dict.getValue<atcg::ref_ptr<Material>>("materia"
                                                                                                               "l"));

    _diffuse_texture   = material->getDiffuseTexture()->getData(atcg::GPU);
    _roughness_texture = material->getRoughnessTexture()->getData(atcg::GPU);
    _ior_texture       = material->getIorTexture()->getData(atcg::GPU);

    DielectricBSDFData data;

    data.diffuse_texture   = TextureSampler<glm::vec3>((std::byte*)_diffuse_texture.data_ptr(),
                                                       material->getDiffuseTexture()->getSpecification());
    data.roughness_texture = TextureSampler<float>((std::byte*)_roughness_texture.data_ptr(),
                                                   material->getRoughnessTexture()->getSpecification());
    data.ior_texture =
        TextureSampler<float>((std::byte*)_ior_texture.data_ptr(), material->getIorTexture()->getSpecification());

    data.optimizable = _optimizable;

    _flags = BSDFComponentType::IdealReflection | BSDFComponentType::IdealReflection;

    _bsdf_data_buffer.upload(&data);
}

DielectricBSDF::~DielectricBSDF() {}

void DielectricBSDF::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                        const atcg::ref_ptr<ShaderBindingTable>& sbt)
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
    uint32_t sample_idx          = sbt->addCallableEntry(sample_prog_group, _bsdf_data_buffer.get());
    uint32_t eval_idx            = sbt->addCallableEntry(eval_prog_group, _bsdf_data_buffer.get());
    uint32_t eval_backward_idx   = sbt->addCallableEntry(backward_eval_prog_group, _bsdf_data_buffer.get());
    uint32_t sample_backward_idx = sbt->addCallableEntry(backward_sample_prog_group, _bsdf_data_buffer.get());
    uint32_t sample_forward_idx  = sbt->addCallableEntry(sample_dual_prog_group, _bsdf_data_buffer.get());
    uint32_t eval_forward_idx    = sbt->addCallableEntry(eval_dual_prog_group, _bsdf_data_buffer.get());

    BSDFVPtrTable table;
    table.sampleCallIndex         = sample_idx;
    table.evalCallIndex           = eval_idx;
    table.evalBackwardCallIndex   = eval_backward_idx;
    table.sampleBackwardCallIndex = sample_backward_idx;
    table.evalForwardCallIndex    = eval_forward_idx;
    table.sampleForwardCallIndex  = sample_forward_idx;
    table.flags                   = _flags;

    _vptr_table.upload(&table);

    markInitialized();
}

std::vector<torch::Tensor> DielectricBSDF::getParameters() const
{
    std::vector<torch::Tensor> parameters;
    if(_optimize_diffuse) parameters.push_back(_diffuse_texture);
    if(_optimize_roughness) parameters.push_back(_roughness_texture);
    if(_optimize_ior) parameters.push_back(_ior_texture);
    return parameters;
}

std::vector<torch::Tensor> DielectricBSDF::getParameterGradients() const
{
    std::vector<torch::Tensor> gradients;
    if(_optimize_diffuse) gradients.push_back(_diffuse_texture_grad);
    if(_optimize_roughness) gradients.push_back(_roughness_texture_grad);
    if(_optimize_ior) gradients.push_back(_ior_texture_grad);
    return gradients;
}

void DielectricBSDF::zeroGrad()
{
    if(_optimize_diffuse) _diffuse_texture_grad.zero_();
    if(_optimize_roughness) _roughness_texture_grad.zero_();
    if(_optimize_ior) _ior_texture_grad.zero_();
}

void DielectricBSDF::onImGuiRender()
{
    if(ImGui::Button("Make Optimizable"))
    {
        markOptimizable();
    }

    if(ImGui::Button("Optimize Diffuse"))
    {
        atcg::TextureSpecification spec_diffuse;
        spec_diffuse.width  = 512;
        spec_diffuse.height = 512;
        spec_diffuse.format = TextureFormat::RGBFLOAT;

        _diffuse_texture = torch::zeros({spec_diffuse.height, spec_diffuse.width, 3},
                                        TensorOptions::floatDeviceOptions().requires_grad(true));

        _diffuse_texture_grad = torch::zeros_like(_diffuse_texture);

        DielectricBSDFData data;
        _bsdf_data_buffer.download(&data);

        data.diffuse_texture = TextureSampler<glm::vec3>((std::byte*)_diffuse_texture.data_ptr(), spec_diffuse);
        data.diffuse_grad    = TextureWriter<glm::vec3>((std::byte*)_diffuse_texture_grad.data_ptr(), spec_diffuse);

        _diffuse_optimized = atcg::Texture2D::create(spec_diffuse);
        _diffuse_grad      = atcg::Texture2D::create(spec_diffuse);

        data.optimize_diffuse = true;
        data.optimizable      = true;

        _bsdf_data_buffer.upload(&data);
    }

    if(ImGui::Button("Optimize Roughness"))
    {
        atcg::TextureSpecification spec_float;
        spec_float.width  = 512;
        spec_float.height = 512;
        spec_float.format = TextureFormat::RFLOAT;

        _roughness_texture = torch::ones({spec_float.height, spec_float.width, 1},
                                         TensorOptions::floatDeviceOptions().requires_grad(true));

        _roughness_texture_grad = torch::zeros_like(_roughness_texture);

        DielectricBSDFData data;
        _bsdf_data_buffer.download(&data);

        data.roughness_texture = TextureSampler<float>((std::byte*)_roughness_texture.data_ptr(), spec_float);
        data.roughness_grad    = TextureWriter<float>((std::byte*)_roughness_texture_grad.data_ptr(), spec_float);

        _roughness_optimized = atcg::Texture2D::create(spec_float);
        _roughness_grad      = atcg::Texture2D::create(spec_float);

        data.optimize_roughness = true;
        data.optimizable        = true;

        _bsdf_data_buffer.upload(&data);
    }

    if(ImGui::Button("Optimize IoR"))
    {
        atcg::TextureSpecification spec_float;
        spec_float.width  = 512;
        spec_float.height = 512;
        spec_float.format = TextureFormat::RFLOAT;

        _ior_texture = torch::full({spec_float.height, spec_float.width, 1},
                                   1.5f,
                                   TensorOptions::floatDeviceOptions().requires_grad(true));

        _ior_texture_grad = torch::zeros_like(_ior_texture);

        DielectricBSDFData data;
        _bsdf_data_buffer.download(&data);

        data.ior_texture = TextureSampler<float>((std::byte*)_ior_texture.data_ptr(), spec_float);
        data.ior_grad    = TextureWriter<float>((std::byte*)_ior_texture_grad.data_ptr(), spec_float);

        _ior_optimized = atcg::Texture2D::create(spec_float);
        _ior_grad      = atcg::Texture2D::create(spec_float);

        data.optimize_ior = true;
        data.optimizable  = true;

        _bsdf_data_buffer.upload(&data);
    }


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


    if(_optimize_diffuse)
    {
        _diffuse_optimized->setData(_diffuse_texture);
        _diffuse_grad->setData(normalize(_diffuse_texture_grad));

        ImGui::Text("Diffuse");
        ImGui::Text("Texture");
        ImGui::Image((ImTextureID)_diffuse_optimized->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
        ImGui::Text("Grad");
        ImGui::Image((ImTextureID)_diffuse_grad->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
        ImGui::Separator();
    }


    if(_optimize_ior)
    {
        _ior_optimized->setData(_ior_texture);
        _ior_grad->setData(pos_neg(_ior_texture_grad));

        ImGui::Text("IoR");
        ImGui::Text("Texture");
        ImGui::Image((ImTextureID)_ior_optimized->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
        ImGui::Text("Grad");
        ImGui::Image((ImTextureID)_ior_grad->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
        ImGui::Separator();
    }

    if(_optimize_roughness)
    {
        _roughness_optimized->setData(_roughness_texture);
        _roughness_grad->setData(pos_neg(_roughness_texture_grad));

        ImGui::Text("Roughness");
        ImGui::Text("Texture");
        ImGui::Image((ImTextureID)_roughness_optimized->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
        ImGui::Text("Grad");
        ImGui::Image((ImTextureID)_roughness_grad->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
        ImGui::Separator();
    }
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

    _diffuse_texture_grad   = torch::zeros_like(_diffuse_texture);
    _ior_texture_grad       = torch::zeros_like(_ior_texture);
    _roughness_texture_grad = torch::zeros_like(_roughness_texture);

    DielectricBSDFData data;

    data.diffuse_texture   = TextureSampler<glm::vec3>((std::byte*)_diffuse_texture.data_ptr(), spec_diffuse);
    data.ior_texture       = TextureSampler<float>((std::byte*)_ior_texture.data_ptr(), spec_float);
    data.roughness_texture = TextureSampler<float>((std::byte*)_roughness_texture.data_ptr(), spec_float);

    data.diffuse_grad   = TextureWriter<glm::vec3>((std::byte*)_diffuse_texture_grad.data_ptr(), spec_diffuse);
    data.ior_grad       = TextureWriter<float>((std::byte*)_ior_texture_grad.data_ptr(), spec_float);
    data.roughness_grad = TextureWriter<float>((std::byte*)_roughness_texture_grad.data_ptr(), spec_float);

    data.optimizable        = true;
    data.optimize_diffuse   = true;
    data.optimize_ior       = true;
    data.optimize_roughness = true;

    _bsdf_data_buffer.upload(&data);

    _diffuse_optimized   = atcg::Texture2D::create(spec_diffuse);
    _ior_optimized       = atcg::Texture2D::create(spec_float);
    _roughness_optimized = atcg::Texture2D::create(spec_float);

    spec_float.format = TextureFormat::RGFLOAT;    // For pos/neg visualization
    _diffuse_grad     = atcg::Texture2D::create(spec_diffuse);
    _ior_grad         = atcg::Texture2D::create(spec_float);
    _roughness_grad   = atcg::Texture2D::create(spec_float);

    _optimize_diffuse   = true;
    _optimize_roughness = true;
    _optimize_ior       = true;
    _optimizable        = true;
}

void DielectricBSDF::clampParameters()
{
    if(_optimize_diffuse) _diffuse_texture.clamp_(0.0f, 1.0f);
    if(_optimize_roughness) _roughness_texture.clamp_(0.0f, 1.0f);
    if(_optimize_ior) _ior_texture.clamp_(1.0f, 2.5f);
}

ATCG_REGISTER_BSDF(MaterialType::MATERIAL_TYPE_DIELECTRIC, DielectricBSDF);
}    // namespace atcg