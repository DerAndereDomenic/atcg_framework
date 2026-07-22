#include <Medium/HenyeyGreensteinPhaseFunction.h>

#ifndef ATCG_HEADLESS
    #include <imgui.h>
    #include <implot.h>
#endif

namespace atcg
{
HenyeyGreensteinPhaseFunction::HenyeyGreensteinPhaseFunction(const atcg::Dictionary& dict) : PhaseFunction(dict)
{
    float g = dict.getValueOr<float>("g", 0.0f);

    auto g_tensor = atcg::createHostTensorFromPointer(&g, {1}).cuda();

    HenyeyGreensteinPhaseFunctionData data;
    data.g = (float*)g_tensor.data_ptr();

    _data_buffer.upload(&data);
    setParameter("g", g_tensor);
}

HenyeyGreensteinPhaseFunction::~HenyeyGreensteinPhaseFunction() {}

void HenyeyGreensteinPhaseFunction::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                                       const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_bsdf_filename = "./bin/HenyeyGreensteinPhaseFunction_ptx.ptx";
    auto sample_prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_hgphase"});
    auto eval_prog_group   = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_hgphase"});
    auto eval_prog_group_forward =
        pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_hgphase_forward"});
    auto eval_backward_prog_group =
        pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_hgphase_backward"});
    auto sample_forward_prog_group =
        pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_hgphase_forward"});
    auto sample_backward_prog_group =
        pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_hgphase_backward"});

    uint32_t sample_idx          = sbt->addCallableEntry(sample_prog_group, _data_buffer.get());
    uint32_t eval_idx            = sbt->addCallableEntry(eval_prog_group, _data_buffer.get());
    uint32_t eval_forward_idx    = sbt->addCallableEntry(eval_prog_group_forward, _data_buffer.get());
    uint32_t eval_backward_idx   = sbt->addCallableEntry(eval_backward_prog_group, _data_buffer.get());
    uint32_t sample_forward_idx  = sbt->addCallableEntry(sample_forward_prog_group, _data_buffer.get());
    uint32_t sample_backward_idx = sbt->addCallableEntry(sample_backward_prog_group, _data_buffer.get());


    PhaseFunctionVPtrTable table;
    table.sampleCallIndex         = sample_idx;
    table.evalCallIndex           = eval_idx;
    table.evalForwardCallIndex    = eval_forward_idx;
    table.evalBackwardCallIndex   = eval_backward_idx;
    table.sampleForwardCallIndex  = sample_forward_idx;
    table.sampleBackwardCallIndex = sample_backward_idx;

    _vptr_table.upload(&table);

    markInitialized();
}

void HenyeyGreensteinPhaseFunction::onImGuiRender()
{
    if(ImGui::Button("Optimize g"))
    {
        auto g_tensor = torch::zeros({1}, atcg::TensorOptions::floatDeviceOptions()).requires_grad_(true);
        setParameter("g", g_tensor);
    }

    if(isParameterOptimizable("g"))
    {
        auto g_tensor      = getParameter("g");
        float density      = g_tensor.item<float>();
        float density_grad = g_tensor.grad().defined() ? g_tensor.grad().item<float>() : 0.0f;

        static int iteration_count = 0;

        time_collection.addSample((float)iteration_count);
        g_collection.addSample(density);
        g_grad_collection.addSample(density_grad);
        iteration_count++;

        if(ImPlot::BeginPlot("g"))
        {
            ImPlot::SetupAxes("Iteration", "g", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotLine("g",
                             time_collection.get(),
                             g_collection.get(),
                             g_collection.count(),
                             0,
                             g_collection.index(),
                             sizeof(float));
            ImPlot::EndPlot();
        }

        if(ImPlot::BeginPlot("g Gradient"))
        {
            ImPlot::SetupAxes("Iteration", "g Gradient", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotLine("g Gradient",
                             time_collection.get(),
                             g_grad_collection.get(),
                             g_grad_collection.count(),
                             0,
                             g_grad_collection.index(),
                             sizeof(float));
            ImPlot::EndPlot();
        }
    }
}

void HenyeyGreensteinPhaseFunction::clampParameters()
{
    if(isParameterOptimizable("g"))
    {
        // Clamp g to [-0.99, 0.99] for stability
        getParameter("g").clamp_(-0.99f, 0.99f);
    }
}

void HenyeyGreensteinPhaseFunction::uploadParameterToDevice(const const std::string& parameter_name)
{
    if(parameter_name == "g")
    {
        auto g_tensor      = getParameter("g");
        auto g_grad_tensor = getGradient("g");

        HenyeyGreensteinPhaseFunctionData data;
        _data_buffer.download(&data);

        data.g          = (float*)g_tensor.data_ptr();
        data.g_grad     = (float*)g_grad_tensor.data_ptr();
        data.optimize_g = g_tensor.requires_grad();

        _data_buffer.upload(&data);
    }
}

}    // namespace atcg