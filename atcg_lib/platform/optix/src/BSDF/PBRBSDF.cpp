#include <BSDF/PBRBSDF.h>

#include <Core/Common.h>
#include <BSDF/BSDFFactory.h>
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
    atcg::ref_ptr<Material> material = dict.getValue<atcg::ref_ptr<Material>>("material");

    _diffuse_texture         = material->getDiffuseTexture()->getData(atcg::GPU);
    _metallic_texture        = material->getMetallicTexture()->getData(atcg::GPU);
    _roughness_texture       = material->getRoughnessTexture()->getData(atcg::GPU);
    _fixed_roughness_texture = _roughness_texture.clone();

    PBRBSDFData data;

    data.diffuse_texture         = TextureSampler<glm::vec3>((std::byte*)_diffuse_texture.data_ptr(),
                                                     material->getDiffuseTexture()->getSpecification());
    data.metallic_texture        = TextureSampler<float>((std::byte*)_metallic_texture.data_ptr(),
                                                  material->getMetallicTexture()->getSpecification());
    data.roughness_texture       = TextureSampler<float>((std::byte*)_roughness_texture.data_ptr(),
                                                   material->getRoughnessTexture()->getSpecification());
    data.fixed_roughness_texture = TextureSampler<float>((std::byte*)_fixed_roughness_texture.data_ptr(),
                                                         material->getRoughnessTexture()->getSpecification());

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

std::vector<torch::Tensor> PBRBSDF::getParameters() const
{
    std::vector<torch::Tensor> parameters;
    if(_diffuse_optimized)
    {
        parameters.push_back(_diffuse_texture);
    }

    if(_metallic_optimized)
    {
        parameters.push_back(_metallic_texture);
    }

    if(_roughness_optimized)
    {
        parameters.push_back(_roughness_texture);
    }

    return parameters;
}

std::vector<torch::Tensor> PBRBSDF::getParameterGradients() const
{
    std::vector<torch::Tensor> gradients;
    if(_diffuse_optimized)
    {
        gradients.push_back(_diffuse_texture_grad);
    }

    if(_metallic_optimized)
    {
        gradients.push_back(_metallic_texture_grad);
    }

    if(_roughness_optimized)
    {
        gradients.push_back(_roughness_texture_grad);
    }

    return gradients;
}

void PBRBSDF::onImGuiRender()
{
    ImGui::SliderInt("Optimization Width", (int*)&_optimization_width, 1, 512);
    ImGui::SliderInt("Optimization Height", (int*)&_optimization_height, 1, 512);

    if(ImGui::Button("Optimize color"))
    {
        atcg::TextureSpecification spec_diffuse;
        spec_diffuse.width  = _optimization_width;
        spec_diffuse.height = _optimization_height;
        spec_diffuse.format = TextureFormat::RGBFLOAT;

        _diffuse_texture = torch::zeros({spec_diffuse.height, spec_diffuse.width, 3},
                                        TensorOptions::floatDeviceOptions().requires_grad(true));

        _diffuse_texture_grad = torch::zeros_like(_diffuse_texture);

        PBRBSDFData data;
        _bsdf_data_buffer.download(&data);

        data.diffuse_texture = TextureSampler<glm::vec3>((std::byte*)_diffuse_texture.data_ptr(), spec_diffuse);
        data.diffuse_grad    = TextureWriter<glm::vec3>((std::byte*)_diffuse_texture_grad.data_ptr(), spec_diffuse);

        _diffuse_optimized = atcg::Texture2D::create(spec_diffuse);
        _diffuse_grad      = atcg::Texture2D::create(spec_diffuse);

        data.optimize_diffuse = true;
        _optimizable          = true;
        _bsdf_data_buffer.upload(&data);
    }

    if(ImGui::Button("Optimize roughness"))
    {
        atcg::TextureSpecification spec_float;
        spec_float.width         = _optimization_width;
        spec_float.height        = _optimization_height;
        spec_float.format        = TextureFormat::RFLOAT;
        _roughness_texture       = torch::full({spec_float.height, spec_float.height, 1},
                                         0.8f,
                                         TensorOptions::floatDeviceOptions().requires_grad(true));    // TODO
        _fixed_roughness_texture = _roughness_texture.clone();

        _roughness_texture_grad = torch::zeros_like(_roughness_texture);

        PBRBSDFData data;
        _bsdf_data_buffer.download(&data);

        data.roughness_texture = TextureSampler<float>((std::byte*)_roughness_texture.data_ptr(), spec_float);
        data.roughness_grad    = TextureWriter<float>((std::byte*)_roughness_texture_grad.data_ptr(), spec_float);
        data.fixed_roughness_texture =
            TextureSampler<float>((std::byte*)_fixed_roughness_texture.data_ptr(), spec_float);

        _roughness_optimized = atcg::Texture2D::create(spec_float);

        spec_float.format = TextureFormat::RGFLOAT;    // For pos/neg visualization
        _roughness_grad   = atcg::Texture2D::create(spec_float);

        data.optimize_roughness = true;
        _optimizable            = true;
        _bsdf_data_buffer.upload(&data);
    }

    if(ImGui::Button("Optimize metallic"))
    {
        atcg::TextureSpecification spec_float;
        spec_float.width  = _optimization_width;
        spec_float.height = _optimization_height;
        spec_float.format = TextureFormat::RFLOAT;

        _metallic_texture = torch::zeros({spec_float.height, spec_float.height, 1},
                                         TensorOptions::floatDeviceOptions().requires_grad(true));

        _metallic_texture_grad = torch::zeros_like(_metallic_texture);

        PBRBSDFData data;
        _bsdf_data_buffer.download(&data);

        data.metallic_texture = TextureSampler<float>((std::byte*)_metallic_texture.data_ptr(), spec_float);
        data.metallic_grad    = TextureWriter<float>((std::byte*)_metallic_texture_grad.data_ptr(), spec_float);

        _metallic_optimized = atcg::Texture2D::create(spec_float);

        spec_float.format = TextureFormat::RGFLOAT;    // For pos/neg visualization
        _metallic_grad    = atcg::Texture2D::create(spec_float);

        data.optimize_metallic = true;
        _optimizable           = true;
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

    if(_diffuse_optimized)
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

    if(_metallic_optimized)
    {
        _metallic_optimized->setData(_metallic_texture);
        _metallic_grad->setData(pos_neg(_metallic_texture_grad));

        ImGui::Text("Metallic");
        ImGui::Text("Texture");
        ImGui::Image((ImTextureID)_metallic_optimized->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
        ImGui::Text("Grad");
        ImGui::Image((ImTextureID)_metallic_grad->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
        ImGui::Separator();
    }

    if(_roughness_optimized)
    {
        _roughness_optimized->setData(_roughness_texture);
        _roughness_grad->setData(
            pos_neg(_roughness_texture.grad().defined() ? _roughness_texture.grad() : _roughness_texture_grad));

        if(_roughness_texture.numel() == 1)
        {
            float roughness = _roughness_texture.cpu().item<float>();
            float roughness_grad =
                _roughness_texture.grad().defined() ? _roughness_texture.grad().cpu().item<float>() : 0.0f;

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

void PBRBSDF::markOptimizable()
{
    atcg::TextureSpecification spec_diffuse;
    spec_diffuse.width  = _optimization_width;
    spec_diffuse.height = _optimization_height;
    spec_diffuse.format = TextureFormat::RGBFLOAT;

    atcg::TextureSpecification spec_float;
    spec_float.width  = _optimization_width;
    spec_float.height = _optimization_height;
    spec_float.format = TextureFormat::RFLOAT;

    _diffuse_texture   = torch::zeros({spec_diffuse.height, spec_diffuse.width, 3},
                                    TensorOptions::floatDeviceOptions().requires_grad(true));
    _metallic_texture  = torch::zeros({spec_float.height, spec_float.height, 1},
                                     TensorOptions::floatDeviceOptions().requires_grad(true));
    _roughness_texture = torch::ones({spec_float.height, spec_float.height, 1},
                                     TensorOptions::floatDeviceOptions().requires_grad(true));    // TODO

    _diffuse_texture_grad   = torch::zeros_like(_diffuse_texture);
    _metallic_texture_grad  = torch::zeros_like(_metallic_texture);
    _roughness_texture_grad = torch::zeros_like(_roughness_texture);

    PBRBSDFData data;

    data.diffuse_texture   = TextureSampler<glm::vec3>((std::byte*)_diffuse_texture.data_ptr(), spec_diffuse);
    data.metallic_texture  = TextureSampler<float>((std::byte*)_metallic_texture.data_ptr(), spec_float);
    data.roughness_texture = TextureSampler<float>((std::byte*)_roughness_texture.data_ptr(), spec_float);

    data.diffuse_grad   = TextureWriter<glm::vec3>((std::byte*)_diffuse_texture_grad.data_ptr(), spec_diffuse);
    data.metallic_grad  = TextureWriter<float>((std::byte*)_metallic_texture_grad.data_ptr(), spec_float);
    data.roughness_grad = TextureWriter<float>((std::byte*)_roughness_texture_grad.data_ptr(), spec_float);

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
    bsdf_data.optimize_diffuse   = true;
    bsdf_data.optimize_metallic  = true;
    bsdf_data.optimize_roughness = true;
    _bsdf_data_buffer.upload(&bsdf_data);
}

void PBRBSDF::clampParameters()
{
    if(!_optimizable) return;
    if(_diffuse_optimized)
    {
        _diffuse_texture.clamp_(0.0f, 1.0f);
    }
    if(_roughness_optimized)
    {
        _roughness_texture.clamp_(0.05f, 1.0f);
    }
    if(_metallic_optimized)
    {
        _metallic_texture.clamp_(0.0f, 1.0f);
    }
    _fixed_roughness_texture.copy_(_roughness_texture);
}

void PBRBSDF::zeroGrad()
{
    if(!_optimizable) return;
    if(_diffuse_optimized)
    {
        _diffuse_texture_grad.zero_();
    }
    if(_roughness_optimized)
    {
        _roughness_texture_grad.zero_();
    }
    if(_metallic_optimized)
    {
        _metallic_texture_grad.zero_();
    }
}

ATCG_REGISTER_BSDF(MaterialType::MATERIAL_TYPE_OPAQUE, PBRBSDF);
}    // namespace atcg