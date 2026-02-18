#include "FiniteDiffPathIntegrator.h"

#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <Utils/Utils.h>

namespace atcg
{
torch::autograd::variable_list FiniteDiffPathNode::apply(torch::autograd::variable_list&& grads)
{
    torch::NoGradGuard no_grad;
    // Apply backward pass for each gradient
    auto parameters = integrator->getParameters();

    std::vector<torch::Tensor> parameter_gradients;

    float h = 0.00001f;

    Dictionary dict;
    dict.setValue("rng_index", rng_index);
    dict.setValue("camera", camera);
    dict.setValue("width", width);
    dict.setValue("height", height);
    for(int i = 0; i < parameters.size(); ++i)
    {
        auto parameter = parameters[i];
        auto copy      = parameter.clone();
        // Perturb the parameter positively
        parameter.copy_(copy + h);
        auto output_pos = integrator->sample(dict);
        parameter.copy_(copy - h);    // Perturb the parameter negatively
        auto output_neg = integrator->sample(dict);
        parameter.copy_(copy);    // Restore original value
        // Compute finite difference gradient
        auto grad = (output_pos - output_neg) / (2 * h);
        parameter_gradients.push_back(torch::sum(grads[0] * grad, {0, 1, 2}, true));
    }

    return parameter_gradients;
}

void FiniteDiffPathNode::release_variables()
{
    camera.reset();
}

FiniteDiffPathtracingIntegrator::FiniteDiffPathtracingIntegrator(const atcg::ref_ptr<RaytracingContext>& context,
                                                                 const Dictionary& dict)
    : DiffPathtracingIntegrator(context, dict)
{
}

FiniteDiffPathtracingIntegrator::~FiniteDiffPathtracingIntegrator() {}

torch::Tensor FiniteDiffPathtracingIntegrator::sample(Dictionary& in_out_dictionary)
{
    const auto& parameters = getParameters();

    bool is_executable = parameters.size() > 0 && torch::autograd::GradMode::is_enabled() &&
                         torch::autograd::any_variable_requires_grad(parameters);

    torch::Tensor result;
    {
        torch::NoGradGuard no_grad;
        result = DiffPathtracingIntegrator::sample(in_out_dictionary);
    }

    if(is_executable)
    {
        std::shared_ptr<FiniteDiffPathNode> node(new FiniteDiffPathNode(), torch::autograd::deleteNode);
        auto next_edges = torch::autograd::collect_next_edges(parameters);
        node->set_next_edges(std::move(next_edges));
        node->integrator = this;
        node->rng_index  = in_out_dictionary.getValueOr<uint32_t>("rng_index", 0);
        node->camera     = in_out_dictionary.getValue<atcg::ref_ptr<atcg::PerspectiveCamera>>("camera");
        node->width      = in_out_dictionary.getValue<uint32_t>("width");
        node->height     = in_out_dictionary.getValue<uint32_t>("height");

        torch::autograd::set_history(result, node);
    }

    return result;
}

}    // namespace atcg