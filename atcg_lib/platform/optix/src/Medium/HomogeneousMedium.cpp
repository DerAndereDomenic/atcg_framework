#include <Medium/HomogeneousMedium.h>

#include <Scene/ComponentRegistry.h>

#ifndef ATCG_HEADLESS
    #include <imgui.h>
    #include <implot.h>
#endif

namespace atcg
{
HomogeneousMedium::HomogeneousMedium(const atcg::Dictionary& dict) : Medium(dict)
{
    glm::vec3 albedo = dict.getValueOr<glm::vec3>("albedo", glm::vec3(0));
    float density    = dict.getValueOr<float>("density", 0.0f);

    auto albedo_tensor  = atcg::createHostTensorFromPointer(glm::value_ptr(albedo), {3}).cuda();
    auto density_tensor = atcg::createHostTensorFromPointer(&density, {1}).cuda();

    setParameter("albedo", albedo_tensor);
    setParameter("density", density_tensor);

    HomogeneousMediumData data;
    data.albedo  = (glm::vec3*)albedo_tensor.data_ptr();
    data.density = (float*)density_tensor.data_ptr();
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
    OptixProgramGroup sample_medium_event_backward_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__homogeneousMedium_sampleMediumEventBackward"});
    OptixProgramGroup eval_transmittance_backward_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__homogeneousMedium_evalTransmittanceBackward"});
    OptixProgramGroup sample_medium_event_forward_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__homogeneousMedium_sampleMediumEventForward"});
    OptixProgramGroup sample_full_backward_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__homogeneousMedium_sampleFullBackward"});
    OptixProgramGroup eval_transmittance_forward_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__homogeneousMedium_evalTransmittanceForward"});

    uint32_t eval_transmittance_index  = sbt->addCallableEntry(eval_transmittance_prog_group, _data_buffer.get());
    uint32_t sample_medium_event_index = sbt->addCallableEntry(sample_medium_event_prog_group, _data_buffer.get());
    uint32_t sample_medium_event_backward_index =
        sbt->addCallableEntry(sample_medium_event_backward_prog_group, _data_buffer.get());
    uint32_t eval_transmittance_backward_index =
        sbt->addCallableEntry(eval_transmittance_backward_prog_group, _data_buffer.get());
    uint32_t sample_medium_event_forward_index =
        sbt->addCallableEntry(sample_medium_event_forward_prog_group, _data_buffer.get());
    uint32_t eval_transmittance_forward_index =
        sbt->addCallableEntry(eval_transmittance_forward_prog_group, _data_buffer.get());
    uint32_t sample_full_backward_index = sbt->addCallableEntry(sample_full_backward_prog_group, _data_buffer.get());

    MediumVPtrTable vptr_table_data;
    vptr_table_data.evalCallIndex                      = eval_transmittance_index;
    vptr_table_data.sampleCallIndex                    = sample_medium_event_index;
    vptr_table_data.sampleBackwardCallIndex            = sample_medium_event_backward_index;
    vptr_table_data.evalTransmittanceForwardCallIndex  = eval_transmittance_forward_index;
    vptr_table_data.phase_function                     = phase_function ? phase_function->getVPtrTable() : nullptr;
    vptr_table_data.evalTransmittanceBackwardCallIndex = eval_transmittance_backward_index;
    vptr_table_data.sampleForwardCallIndex             = sample_medium_event_forward_index;
    vptr_table_data.sampleFullBackwardCallIndex        = sample_full_backward_index;

    _vptr_table.upload(&vptr_table_data);

    markInitialized();
}

void HomogeneousMedium::onImGuiRender()
{
    if(ImGui::Button("Optimize Albedo"))
    {
        markParameterAsOptimizable("albedo");
    }

    if(ImGui::Button("Optimize Density"))
    {
        markParameterAsOptimizable("density");
    }

    if(isParameterOptimizable("density"))
    {
        auto density_tensor = getParameter("density");
        float density       = density_tensor.item<float>();
        float density_grad  = density_tensor.grad().defined() ? density_tensor.grad().item<float>() : 0.0f;

        static int iteration_count = 0;

        time_collection.addSample((float)iteration_count);
        density_collection.addSample(density);
        density_grad_collection.addSample(density_grad);
        iteration_count++;

        if(ImPlot::BeginPlot("Density"))
        {
            ImPlot::SetupAxes("Iteration", "Density", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotLine("Density",
                             time_collection.get(),
                             density_collection.get(),
                             density_collection.count(),
                             0,
                             density_collection.index(),
                             sizeof(float));
            ImPlot::EndPlot();
        }

        if(ImPlot::BeginPlot("Density Gradient"))
        {
            ImPlot::SetupAxes("Iteration", "Density Gradient", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotLine("Density Gradient",
                             time_collection.get(),
                             density_grad_collection.get(),
                             density_grad_collection.count(),
                             0,
                             density_grad_collection.index(),
                             sizeof(float));
            ImPlot::EndPlot();
        }
    }
}

void HomogeneousMedium::clampParameters()
{
    if(isParameterOptimizable("albedo"))
    {
        auto albedo_tensor = getParameter("albedo");
        albedo_tensor.clamp_(0.0f, 1.0f);
    }

    if(isParameterOptimizable("density"))
    {
        auto density_tensor = getParameter("density");
        density_tensor.clamp_(0.0f, std::numeric_limits<float>::max());    // TODO
    }
}

void HomogeneousMedium::markParameterAsOptimizable(const std::string& parameter_name)
{
    if(parameter_name == "albedo")
    {
        auto albedo_tensor = torch::ones({3}, atcg::TensorOptions::floatDeviceOptions()).requires_grad_(true);
        setParameter("albedo", albedo_tensor);
        auto albedo_grad_tensor = getGradient("albedo");

        HomogeneousMediumData data;
        _data_buffer.download(&data);

        data.albedo = (glm::vec3*)albedo_tensor.data_ptr();

        data.albedo_grad = (float*)albedo_grad_tensor.data_ptr();

        data.optimize_albedo = true;

        _data_buffer.upload(&data);
    }
    else if(parameter_name == "density")
    {
        auto density_tensor = torch::full({1}, 0.5f, atcg::TensorOptions::floatDeviceOptions()).requires_grad_(true);
        setParameter("density", density_tensor);
        auto density_grad_tensor = getGradient("density");

        HomogeneousMediumData data;
        _data_buffer.download(&data);

        data.density          = (float*)density_tensor.data_ptr();
        data.density_grad     = (float*)density_grad_tensor.data_ptr();
        data.optimize_density = true;

        _data_buffer.upload(&data);
    }
}

}    // namespace atcg