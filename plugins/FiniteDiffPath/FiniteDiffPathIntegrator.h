#pragma once

#include <Integrator/DifferentiableIntegrator.h>

#include <torch/torch.h>

namespace atcg
{
struct FiniteDiffPathNode : public torch::autograd::Node
{
    DifferentiableIntegrator* integrator;
    uint32_t rng_index;
    uint32_t width, height;
    torch::autograd::variable_list apply(torch::autograd::variable_list&& grads) override;

    virtual void release_variables() override;
};

class FiniteDiffPathtracingIntegrator : public DifferentiableIntegrator
{
public:
    FiniteDiffPathtracingIntegrator(const atcg::ref_ptr<RaytracingContext>& context, const Dictionary& dict);

    ~FiniteDiffPathtracingIntegrator();

    virtual void onImGuiRender() override;

    virtual torch::Tensor sample(Dictionary& in_out_dictionary) override;

    virtual void reset() override;

    virtual std::vector<torch::Tensor> getParameters() const override;

    virtual std::vector<torch::Tensor> getParameterGradients() const override;

    virtual void clampParameters() override;

    virtual void markOptimizable() override;

    virtual void zeroGrad() override;

private:
    friend class FiniteDiffPathNode;

    atcg::ref_ptr<DifferentiableIntegrator> _integrator;
};
}    // namespace atcg