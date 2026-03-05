#include <Medium/HomogeneousMedium.h>

#include <Scene/ComponentRegistry.h>

#ifndef ATCG_HEADLESS
    #include <imgui.h>
#endif

namespace atcg
{
HomogeneousMedium::HomogeneousMedium(const atcg::Dictionary& dict) : Medium(dict)
{
    glm::vec3 albedo = dict.getValueOr<glm::vec3>("albedo", glm::vec3(0));
    float density    = dict.getValueOr<float>("density", 0.0f);

    _albedo_tensor  = atcg::createHostTensorFromPointer(glm::value_ptr(albedo), {3}).cuda();
    _density_tensor = atcg::createHostTensorFromPointer(&density, {1}).cuda();

    HomogeneousMediumData data;
    data.albedo  = (glm::vec3*)_albedo_tensor.data_ptr();
    data.density = (float*)_density_tensor.data_ptr();
    data.Le      = glm::vec3(dict.getValueOr<glm::vec3>("Le", glm::vec3(0)));

    _data_buffer.upload(&data);
}

HomogeneousMedium::~HomogeneousMedium() {}

void HomogeneousMedium::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                           const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    if(_phase_function != nullptr) _phase_function->ensureInitialized(pipeline, sbt);

    auto phase_function = getPhaseFunction();

    const std::string ptx_filename = "./bin/HomogeneousMedium_ptx.ptx";
    OptixProgramGroup eval_transmittance_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__homogeneousMedium_evalTransmittance"});
    OptixProgramGroup sample_medium_event_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__homogeneousMedium_sampleMediumEvent"});

    uint32_t eval_transmittance_index  = sbt->addCallableEntry(eval_transmittance_prog_group, _data_buffer.get());
    uint32_t sample_medium_event_index = sbt->addCallableEntry(sample_medium_event_prog_group, _data_buffer.get());

    MediumVPtrTable vptr_table_data;
    vptr_table_data.evalCallIndex   = eval_transmittance_index;
    vptr_table_data.sampleCallIndex = sample_medium_event_index;
    vptr_table_data.phase_function  = phase_function ? phase_function->getVPtrTable() : nullptr;

    _vptr_table.upload(&vptr_table_data);

    markInitialized();
}

void HomogeneousMedium::onImGuiRender()
{
    if(ImGui::Button("Optimize Albedo"))
    {
        _albedo_tensor = torch::ones({3}, atcg::TensorOptions::floatDeviceOptions()).requires_grad_(true);


        _albedo_grad_tensor = torch::zeros({3}, atcg::TensorOptions::floatDeviceOptions());

        HomogeneousMediumData data;
        _data_buffer.download(&data);

        data.albedo = (glm::vec3*)_albedo_tensor.data_ptr();

        data.albdeo_grad = (glm::vec3*)_albedo_grad_tensor.data_ptr();

        data.optimize_albedo = true;

        _data_buffer.upload(&data);

        _optimizable     = true;
        _optimize_albedo = true;
    }

    if(ImGui::Button("Optimize Density"))
    {
        _density_tensor      = torch::ones({1}, atcg::TensorOptions::floatDeviceOptions()).requires_grad_(true);
        _density_grad_tensor = torch::zeros({1}, atcg::TensorOptions::floatDeviceOptions());

        HomogeneousMediumData data;
        _data_buffer.download(&data);

        data.density          = (float*)_density_tensor.data_ptr();
        data.density_grad     = (float*)_density_grad_tensor.data_ptr();
        data.optimize_density = true;

        _data_buffer.upload(&data);

        _optimizable      = true;
        _optimize_density = true;
    }
}

std::vector<torch::Tensor> HomogeneousMedium::getParameters() const
{
    std::vector<torch::Tensor> params;
    if(_optimize_albedo) params.push_back(_albedo_tensor);
    if(_optimize_density) params.push_back(_density_tensor);
    return params;
}

std::vector<torch::Tensor> HomogeneousMedium::getParameterGradients() const
{
    std::vector<torch::Tensor> grads;
    if(_optimize_albedo) grads.push_back(_albedo_grad_tensor);
    if(_optimize_density) grads.push_back(_density_grad_tensor);
    return grads;
}

void HomogeneousMedium::zeroGrad()
{
    if(_optimize_albedo) _albedo_grad_tensor.zero_();
    if(_optimize_density) _density_grad_tensor.zero_();
}

void HomogeneousMedium::markOptimizable()
{
    _albedo_tensor  = torch::ones({3}, atcg::TensorOptions::floatDeviceOptions()).requires_grad_(true);
    _density_tensor = torch::ones({1}, atcg::TensorOptions::floatDeviceOptions()).requires_grad_(true);

    _albedo_grad_tensor  = torch::zeros({3}, atcg::TensorOptions::floatDeviceOptions());
    _density_grad_tensor = torch::zeros({1}, atcg::TensorOptions::floatDeviceOptions());

    HomogeneousMediumData data;
    _data_buffer.download(&data);

    data.albedo  = (glm::vec3*)_albedo_tensor.data_ptr();
    data.density = (float*)_density_tensor.data_ptr();

    data.albdeo_grad  = (glm::vec3*)_albedo_grad_tensor.data_ptr();
    data.density_grad = (float*)_density_grad_tensor.data_ptr();

    data.optimize_albedo  = true;
    data.optimize_density = true;

    _data_buffer.upload(&data);

    _optimizable      = true;
    _optimize_albedo  = true;
    _optimize_density = true;
}

void HomogeneousMedium::clampParameters()
{
    if(_optimize_albedo)
    {
        _albedo_tensor.clamp_(0.0f, 1.0f);
    }

    if(_optimize_density)
    {
        _density_tensor.clamp_(0.1f, std::numeric_limits<float>::max());    // TODO
    }
}
}    // namespace atcg