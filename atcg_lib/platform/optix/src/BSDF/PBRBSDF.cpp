#include <BSDF/PBRBSDF.h>

#include <Core/Common.h>
#include <Renderer/Texture.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <c10/cuda/CUDAGuard.h>

#ifndef ATCG_HEADLESS
    #include <implot.h>
#endif

namespace atcg
{

PBRBSDF::PBRBSDF(const Dictionary& dict)
{
    atcg::ref_ptr<OpaqueMaterial> material =
        std::dynamic_pointer_cast<atcg::OpaqueMaterial>(dict.getValue<atcg::ref_ptr<Material>>("material"));

    torch::Tensor diffuse_texture   = material->getDiffuseTexture()->getData(atcg::GPU);
    torch::Tensor metallic_texture  = material->getMetallicTexture()->getData(atcg::GPU);
    torch::Tensor roughness_texture = material->getRoughnessTexture()->getData(atcg::GPU);

    setParameter("diffuse_texture", diffuse_texture);
    setParameter("metallic_texture", metallic_texture);
    setParameter("roughness_texture", roughness_texture);

    PBRBSDFData data;

    data.diffuse_texture   = TextureSampler<glm::vec3>((std::byte*)diffuse_texture.data_ptr(),
                                                       material->getDiffuseTexture()->getSpecification());
    data.metallic_texture  = TextureSampler<float>((std::byte*)metallic_texture.data_ptr(),
                                                   material->getMetallicTexture()->getSpecification());
    data.roughness_texture = TextureSampler<float>((std::byte*)roughness_texture.data_ptr(),
                                                   material->getRoughnessTexture()->getSpecification());
    // data.fixed_roughness_texture = TextureSampler<float>((std::byte*)fixed_roughness_texture.data_ptr(),
    //                                                      material->getRoughnessTexture()->getSpecification());

    _flags = BSDFComponentType::GlossyReflection | BSDFComponentType::DiffuseReflection;

    _bsdf_data_buffer.upload(&data);
}

PBRBSDF::~PBRBSDF() {}

void PBRBSDF::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                 const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_bsdf_filename = "./bin/PBRBSDF_ptx.ptx";

    auto sample_prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_pbrbsdf"});
    auto eval_prog_group   = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_pbrbsdf"});
    auto backward_eval_prog_group =
        pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_backward_pbrbsdf"});
    auto backward_sample_prog_group =
        pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_backward_pbrbsdf"});
    auto eval_dual_prog_group =
        pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_forward_pbrbsdf"});
    auto sample_dual_prog_group =
        pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_forward_pbrbsdf"});
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

void PBRBSDF::onImGuiRender()
{
    ImGui::SliderInt("Optimization Width", (int*)&_optimization_width, 1, 512);
    ImGui::SliderInt("Optimization Height", (int*)&_optimization_height, 1, 512);

    if(ImGui::Button("Optimize color"))
    {
        markParameterAsOptimizable("diffuse_texture");
    }

    if(ImGui::Button("Optimize roughness"))
    {
        markParameterAsOptimizable("roughness_texture");
    }

    if(ImGui::Button("Optimize metallic"))
    {
        markParameterAsOptimizable("metallic_texture");
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

    if(isParameterOptimizable("metallic_texture"))
    {
        auto metallic_texture      = getParameter("metallic_texture");
        auto metallic_texture_grad = getGradient("metallic_texture");
        _metallic_optimized->setData(metallic_texture);
        _metallic_grad->setData(pos_neg(metallic_texture_grad));

        ImGui::Text("Metallic");
        ImGui::Text("Texture");
        ImGui::Image((ImTextureID)_metallic_optimized->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
        ImGui::Text("Grad");
        ImGui::Image((ImTextureID)_metallic_grad->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
        ImGui::Separator();
    }

    if(isParameterOptimizable("roughness_texture"))
    {
        auto roughness_texture      = getParameter("roughness_texture");
        auto roughness_texture_grad = getGradient("roughness_texture");
        _roughness_optimized->setData(roughness_texture);
        _roughness_grad->setData(pos_neg(roughness_texture_grad));

        if(roughness_texture.numel() == 1)
        {
            float roughness = roughness_texture.cpu().item<float>();
            float roughness_grad =
                roughness_texture.grad().defined() ? roughness_texture.grad().cpu().item<float>() : 0.0f;

            static int iteration_count = 0;
            time_collection.addSample((float)iteration_count);
            roughness_collection.addSample(roughness);
            roughness_grad_collection.addSample(roughness_grad);
            iteration_count++;

            if(ImPlot::BeginPlot("Roughness"))
            {
                ImPlot::SetupAxes("Iteration", "Roughness", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                ImPlot::PlotLine("Roughness",
                                 time_collection.get(),
                                 roughness_collection.get(),
                                 roughness_collection.count(),
                                 0,
                                 roughness_collection.index(),
                                 sizeof(float));
                ImPlot::EndPlot();
            }

            if(ImPlot::BeginPlot("Roughness Gradient"))
            {
                ImPlot::SetupAxes("Iteration", "Roughness Gradient", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                ImPlot::PlotLine("Roughness Gradient",
                                 time_collection.get(),
                                 roughness_grad_collection.get(),
                                 roughness_grad_collection.count(),
                                 0,
                                 roughness_grad_collection.index(),
                                 sizeof(float));
                ImPlot::EndPlot();
            }
        }

        ImGui::Text("Roughness");
        ImGui::Text("Texture");
        ImGui::Image((ImTextureID)_roughness_optimized->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
        ImGui::Text("Grad");
        ImGui::Image((ImTextureID)_roughness_grad->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
        ImGui::Separator();
    }
}

void PBRBSDF::clampParameters()
{
    if(isParameterOptimizable("diffuse_texture"))
    {
        auto& diffuse_texture = getParameter("diffuse_texture");
        diffuse_texture.clamp_(0.0f, 1.0f);
    }
    if(isParameterOptimizable("roughness_texture"))
    {
        auto& roughness_texture = getParameter("roughness_texture");
        roughness_texture.clamp_(0.05f, 1.0f);
    }
    if(isParameterOptimizable("metallic_texture"))
    {
        auto& metallic_texture = getParameter("metallic_texture");
        metallic_texture.clamp_(0.0f, 1.0f);
    }
    // _fixed_roughness_texture.copy_(_roughness_texture);
}

void PBRBSDF::markParameterAsOptimizable(const const std::string& parameter_name)
{
    if(parameter_name == "diffuse_texture")
    {
        atcg::TextureSpecification spec_diffuse;
        spec_diffuse.width  = _optimization_width;
        spec_diffuse.height = _optimization_height;
        spec_diffuse.format = TextureFormat::RGBFLOAT;

        auto diffuse_texture = torch::zeros({_optimization_height, _optimization_width, 3},
                                            TensorOptions::floatDeviceOptions().requires_grad(true));
        setParameter("diffuse_texture", diffuse_texture);
        auto diffuse_texture_grad = getGradient("diffuse_texture");

        PBRBSDFData data;
        _bsdf_data_buffer.download(&data);

        data.diffuse_texture = TextureSampler<glm::vec3>((std::byte*)diffuse_texture.data_ptr(), spec_diffuse);
        data.diffuse_grad    = TextureWriter<glm::vec3>((std::byte*)diffuse_texture_grad.data_ptr(), spec_diffuse);

        _diffuse_optimized = atcg::Texture2D::create(spec_diffuse);
        _diffuse_grad      = atcg::Texture2D::create(spec_diffuse);

        data.optimize_diffuse = true;
        _bsdf_data_buffer.upload(&data);
    }

    if(parameter_name == "roughness_texture")
    {
        atcg::TextureSpecification spec_float;
        spec_float.width  = _optimization_width;
        spec_float.height = _optimization_height;
        spec_float.format = TextureFormat::RFLOAT;

        auto roughness_texture = torch::full({_optimization_height, _optimization_width, 1},
                                             0.8f,
                                             TensorOptions::floatDeviceOptions().requires_grad(true));
        setParameter("roughness_texture", roughness_texture);
        auto roughness_texture_grad = getGradient("roughness_texture");

        PBRBSDFData data;
        _bsdf_data_buffer.download(&data);

        data.roughness_texture = TextureSampler<float>((std::byte*)roughness_texture.data_ptr(), spec_float);
        data.roughness_grad    = TextureWriter<float>((std::byte*)roughness_texture_grad.data_ptr(), spec_float);
        // data.fixed_roughness_texture =
        //     TextureSampler<float>((std::byte*)_fixed_roughness_texture.data_ptr(), spec_float);

        _roughness_optimized = atcg::Texture2D::create(spec_float);

        spec_float.format = TextureFormat::RGFLOAT;    // For pos/neg visualization
        _roughness_grad   = atcg::Texture2D::create(spec_float);

        data.optimize_roughness = true;
        _bsdf_data_buffer.upload(&data);
    }

    if(parameter_name == "metallic_texture")
    {
        atcg::TextureSpecification spec_float;
        spec_float.width  = _optimization_width;
        spec_float.height = _optimization_height;
        spec_float.format = TextureFormat::RFLOAT;

        auto metallic_texture = torch::zeros({_optimization_height, _optimization_width, 1},
                                             TensorOptions::floatDeviceOptions().requires_grad(true));
        setParameter("metallic_texture", metallic_texture);
        auto metallic_texture_grad = getGradient("metallic_texture");

        PBRBSDFData data;
        _bsdf_data_buffer.download(&data);

        data.metallic_texture = TextureSampler<float>((std::byte*)metallic_texture.data_ptr(), spec_float);
        data.metallic_grad    = TextureWriter<float>((std::byte*)metallic_texture_grad.data_ptr(), spec_float);

        _metallic_optimized = atcg::Texture2D::create(spec_float);

        spec_float.format = TextureFormat::RGFLOAT;    // For pos/neg visualization
        _metallic_grad    = atcg::Texture2D::create(spec_float);

        data.optimize_metallic = true;
        _bsdf_data_buffer.upload(&data);
    }
}


void PBRBSDF::registerBSDF(BSDFRegistry::Registry* registry)
{
    ATCG_REGISTER_BSDF(registry, "Opaque", PBRBSDF);
}
}    // namespace atcg