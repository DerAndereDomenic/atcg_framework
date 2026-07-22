#include <BSDF/DielectricBSDF.h>

#include <Renderer/Texture.h>
#include <Renderer/Material.h>

#include <Core/Common.h>

namespace atcg
{

DielectricBSDF::DielectricBSDF(const Dictionary& dict)
{
    auto material = std::dynamic_pointer_cast<atcg::DielectricMaterial>(dict.getValue<atcg::ref_ptr<Material>>("materia"
                                                                                                               "l"));

    auto diffuse_texture   = material->getDiffuseTexture()->getData(atcg::GPU);
    auto roughness_texture = material->getRoughnessTexture()->getData(atcg::GPU);
    auto ior_texture       = material->getIorTexture()->getData(atcg::GPU);

    setParameter("diffuse_texture", diffuse_texture);
    setParameter("roughness_texture", roughness_texture);
    setParameter("ior_texture", ior_texture);

    DielectricBSDFData data;

    data.diffuse_texture   = TextureSampler<glm::vec3>((std::byte*)diffuse_texture.data_ptr(),
                                                       material->getDiffuseTexture()->getSpecification());
    data.roughness_texture = TextureSampler<float>((std::byte*)roughness_texture.data_ptr(),
                                                   material->getRoughnessTexture()->getSpecification());
    data.ior_texture =
        TextureSampler<float>((std::byte*)ior_texture.data_ptr(), material->getIorTexture()->getSpecification());

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

void DielectricBSDF::onImGuiRender()
{
    if(ImGui::Button("Optimize Diffuse"))
    {
        markParameterAsOptimizable("diffuse_texture");
    }

    if(ImGui::Button("Optimize Roughness"))
    {
        markParameterAsOptimizable("roughness_texture");
    }

    if(ImGui::Button("Optimize IoR"))
    {
        markParameterAsOptimizable("ior_texture");
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


    if(isParameterOptimizable("diffuse_texture"))
    {
        auto diffuse_texture      = getParameter("diffuse_texture");
        auto diffuse_texture_grad = getGradient("diffuse_texture");

        _diffuse_optimized->setData(diffuse_texture);
        _diffuse_grad->setData(normalize(diffuse_texture_grad));

        ImGui::Text("Diffuse");
        ImGui::Text("Texture");
        ImGui::Image((ImTextureID)_diffuse_optimized->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
        ImGui::Text("Grad");
        ImGui::Image((ImTextureID)_diffuse_grad->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
        ImGui::Separator();
    }


    if(isParameterOptimizable("ior_texture"))
    {
        auto ior_texture      = getParameter("ior_texture");
        auto ior_texture_grad = getGradient("ior_texture");

        _ior_optimized->setData(ior_texture);
        _ior_grad->setData(pos_neg(ior_texture_grad));

        ImGui::Text("IoR");
        ImGui::Text("Texture");
        ImGui::Image((ImTextureID)_ior_optimized->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
        ImGui::Text("Grad");
        ImGui::Image((ImTextureID)_ior_grad->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
        ImGui::Separator();
    }

    if(isParameterOptimizable("roughness_texture"))
    {
        auto roughness_texture      = getParameter("roughness_texture");
        auto roughness_texture_grad = getGradient("roughness_texture");

        _roughness_optimized->setData(roughness_texture);
        _roughness_grad->setData(pos_neg(roughness_texture_grad));

        ImGui::Text("Roughness");
        ImGui::Text("Texture");
        ImGui::Image((ImTextureID)_roughness_optimized->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
        ImGui::Text("Grad");
        ImGui::Image((ImTextureID)_roughness_grad->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
        ImGui::Separator();
    }
}

void DielectricBSDF::clampParameters()
{
    if(isParameterOptimizable("diffuse_texture"))
    {
        auto diffuse_texture = getParameter("diffuse_texture");
        diffuse_texture.clamp_(0.0f, 1.0f);
    }
    if(isParameterOptimizable("roughness_texture"))
    {
        auto roughness_texture = getParameter("roughness_texture");
        roughness_texture.clamp_(0.0f, 1.0f);
    }
    if(isParameterOptimizable("ior_texture"))
    {
        auto ior_texture = getParameter("ior_texture");
        ior_texture.clamp_(1.0f, 2.5f);
    }
}

void DielectricBSDF::markParameterAsOptimizable(const const std::string& parameter_name)
{
    if(parameter_name == "diffuse_texture")
    {
        atcg::TextureSpecification spec_diffuse;
        spec_diffuse.width   = 512;
        spec_diffuse.height  = 512;
        spec_diffuse.format  = TextureFormat::RGBFLOAT;
        auto diffuse_texture = torch::zeros({512, 512, 3}, TensorOptions::floatDeviceOptions().requires_grad(true));
        setParameter("diffuse_texture", diffuse_texture);
        auto diffuse_texture_grad = getGradient("diffuse_texture");

        DielectricBSDFData data;
        _bsdf_data_buffer.download(&data);

        data.diffuse_texture = TextureSampler<glm::vec3>((std::byte*)diffuse_texture.data_ptr(), spec_diffuse);
        data.diffuse_grad    = TextureWriter<glm::vec3>((std::byte*)diffuse_texture_grad.data_ptr(), spec_diffuse);

        _diffuse_optimized = atcg::Texture2D::create(spec_diffuse);
        _diffuse_grad      = atcg::Texture2D::create(spec_diffuse);

        data.optimize_diffuse = true;

        _bsdf_data_buffer.upload(&data);
    }
    else if(parameter_name == "roughness_texture")
    {
        atcg::TextureSpecification spec_float;
        spec_float.width       = 512;
        spec_float.height      = 512;
        spec_float.format      = TextureFormat::RFLOAT;
        auto roughness_texture = torch::ones({512, 512, 1}, TensorOptions::floatDeviceOptions().requires_grad(true));
        setParameter("roughness_texture", roughness_texture);
        auto roughness_texture_grad = getGradient("roughness_texture");

        DielectricBSDFData data;
        _bsdf_data_buffer.download(&data);

        data.roughness_texture = TextureSampler<float>((std::byte*)roughness_texture.data_ptr(), spec_float);
        data.roughness_grad    = TextureWriter<float>((std::byte*)roughness_texture_grad.data_ptr(), spec_float);

        _roughness_optimized = atcg::Texture2D::create(spec_float);
        _roughness_grad      = atcg::Texture2D::create(spec_float);

        data.optimize_roughness = true;

        _bsdf_data_buffer.upload(&data);
    }
    else if(parameter_name == "ior_texture")
    {
        atcg::TextureSpecification spec_float;
        spec_float.width  = 512;
        spec_float.height = 512;
        spec_float.format = TextureFormat::RFLOAT;

        auto ior_texture = torch::full({512, 512, 1}, 1.5f, TensorOptions::floatDeviceOptions().requires_grad(true));
        setParameter("ior_texture", ior_texture);
        auto ior_texture_grad = getGradient("ior_texture");

        DielectricBSDFData data;
        _bsdf_data_buffer.download(&data);

        data.ior_texture = TextureSampler<float>((std::byte*)ior_texture.data_ptr(), spec_float);
        data.ior_grad    = TextureWriter<float>((std::byte*)ior_texture_grad.data_ptr(), spec_float);

        _ior_optimized = atcg::Texture2D::create(spec_float);
        _ior_grad      = atcg::Texture2D::create(spec_float);

        data.optimize_ior = true;

        _bsdf_data_buffer.upload(&data);
    }
}

void DielectricBSDF::registerBSDF(BSDFRegistry::Registry* registry)
{
    ATCG_REGISTER_BSDF(registry, "Dielectric", DielectricBSDF);
}
}    // namespace atcg