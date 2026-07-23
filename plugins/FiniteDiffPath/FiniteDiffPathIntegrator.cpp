#include "FiniteDiffPathIntegrator.h"

#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <Utils/Utils.h>
#include <Scene/OptixScene.h>

namespace atcg
{
torch::autograd::variable_list FiniteDiffPathNode::apply(torch::autograd::variable_list&& grads)
{
    torch::NoGradGuard no_grad;
    // Apply backward pass for each gradient
    auto optix_scene = integrator->getDictionary().getValue<atcg::ref_ptr<OptixScene>>("optix_scene");
    auto parameters  = optix_scene->getParameters();

    std::vector<torch::Tensor> parameter_gradients;

    float h = 0.001f;

    Dictionary dict;
    dict.setValue("rng_index", rng_index);
    for(int i = 0; i < parameters.size(); ++i)
    {
        auto parameter = parameters[i];
        auto copy      = parameter.clone();
        // Perturb the parameter positively
        parameter.copy_(copy + h);
        Dictionary dict;
        integrator->generateRays(dict);
        torch::Tensor output_pos = dict.getValue<torch::Tensor>("output_img");
        parameter.copy_(copy - h);    // Perturb the parameter negatively
        integrator->generateRays(dict);
        torch::Tensor output_neg = dict.getValue<torch::Tensor>("output_img");
        parameter.copy_(copy);    // Restore original value
        // Compute finite difference gradient
        auto grad = (output_pos - output_neg) / (2 * h);
        parameter_gradients.push_back(torch::sum(grads[0] * grad, {0, 1, 2}, true));
    }

    return parameter_gradients;
}

void FiniteDiffPathNode::release_variables() {}

FiniteDiffPathtracingIntegrator::FiniteDiffPathtracingIntegrator(const atcg::ref_ptr<RaytracingContext>& context,
                                                                 const Dictionary& dict)
    : Integrator(context, dict)
{
    _integrator = dict.getValue<atcg::ref_ptr<Integrator>>("integrator");
}

FiniteDiffPathtracingIntegrator::~FiniteDiffPathtracingIntegrator() {}

void FiniteDiffPathtracingIntegrator::generateRays(Dictionary& in_out_dictionary)
{
    auto optix_scene       = _integrator->getDictionary().getValue<atcg::ref_ptr<OptixScene>>("optix_scene");
    const auto& parameters = optix_scene->getParameters();

    bool is_executable = parameters.size() > 0 && torch::autograd::GradMode::is_enabled() &&
                         torch::autograd::any_variable_requires_grad(parameters);

    torch::Tensor result;
    {
        torch::NoGradGuard no_grad;
        _integrator->generateRays(in_out_dictionary);
        result = in_out_dictionary.getValue<torch::Tensor>("output_img");
    }

    if(is_executable)
    {
        std::shared_ptr<FiniteDiffPathNode> node(new FiniteDiffPathNode(), torch::autograd::deleteNode);
        auto next_edges = torch::autograd::collect_next_edges(parameters);
        node->set_next_edges(std::move(next_edges));
        node->integrator = _integrator.get();
        node->rng_index  = in_out_dictionary.getValueOr<uint32_t>("rng_index", 0);

        torch::autograd::set_history(result, node);
    }
}

void FiniteDiffPathtracingIntegrator::onImGuiRender()
{
    _integrator->onImGuiRender();
}

void FiniteDiffPathtracingIntegrator::reset()
{
    _integrator->reset();
}

}    // namespace atcg