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

    _g_tensor = atcg::createHostTensorFromPointer(&g, {1}).cuda();

    HenyeyGreensteinPhaseFunctionData data;
    data.g = (float*)_g_tensor.data_ptr();

    _data_buffer.upload(&data);
}

HenyeyGreensteinPhaseFunction::~HenyeyGreensteinPhaseFunction() {}

void HenyeyGreensteinPhaseFunction::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                                       const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_bsdf_filename = "./bin/HenyeyGreensteinPhaseFunction_ptx.ptx";
    auto sample_prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_hgphase"});
    auto eval_prog_group   = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_hgphase"});
    auto eval_backward_prog_group =
        pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_hgphase_backward"});
    uint32_t sample_idx        = sbt->addCallableEntry(sample_prog_group, _data_buffer.get());
    uint32_t eval_idx          = sbt->addCallableEntry(eval_prog_group, _data_buffer.get());
    uint32_t eval_backward_idx = sbt->addCallableEntry(eval_backward_prog_group, _data_buffer.get());

    PhaseFunctionVPtrTable table;
    table.sampleCallIndex       = sample_idx;
    table.evalCallIndex         = eval_idx;
    table.evalBackwardCallIndex = eval_backward_idx;

    _vptr_table.upload(&table);

    markInitialized();
}

void HenyeyGreensteinPhaseFunction::onImGuiRender()
{
    if(ImGui::Button("Optimize g"))
    {
        _g_tensor = torch::zeros({1}, atcg::TensorOptions::floatDeviceOptions()).requires_grad_(true);

        _g_grad_tensor = torch::zeros({1}, atcg::TensorOptions::floatDeviceOptions());

        HenyeyGreensteinPhaseFunctionData data;
        _data_buffer.download(&data);

        data.g          = (float*)_g_tensor.data_ptr();
        data.g_grad     = (float*)_g_grad_tensor.data_ptr();
        data.optimize_g = true;

        _data_buffer.upload(&data);

        _optimize_g  = true;
        _optimizable = true;
    }

    if(_optimize_g)
    {
        float density      = _g_tensor.item<float>();
        float density_grad = _g_tensor.grad().defined() ? _g_tensor.grad().item<float>() : 0.0f;

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

std::vector<torch::Tensor> HenyeyGreensteinPhaseFunction::getParameters() const
{
    std::vector<torch::Tensor> params;

    if(_optimize_g) params.push_back(_g_tensor);

    return params;
}

std::vector<torch::Tensor> HenyeyGreensteinPhaseFunction::getParameterGradients() const
{
    std::vector<torch::Tensor> gradients;

    if(_optimize_g) gradients.push_back(_g_grad_tensor);

    return gradients;
}

void HenyeyGreensteinPhaseFunction::zeroGrad()
{
    if(_optimize_g) _g_grad_tensor.zero_();
}

void HenyeyGreensteinPhaseFunction::markOptimizable()
{
    // TODO
}

void HenyeyGreensteinPhaseFunction::clampParameters()
{
    if(_optimize_g)
    {
        // Clamp g to [-0.99, 0.99] for stability
        _g_tensor.data().clamp_(-0.99f, 0.99f);
    }
}
}    // namespace atcg