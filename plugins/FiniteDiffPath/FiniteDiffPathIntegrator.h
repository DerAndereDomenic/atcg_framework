#pragma once

#include <torch/torch.h>
#include <Integrator/Integrator.h>

namespace atcg
{
struct FiniteDiffPathNode : public torch::autograd::Node
{
    Integrator* integrator;
    uint32_t rng_index;
    uint32_t width, height;
    torch::autograd::variable_list apply(torch::autograd::variable_list&& grads) override;

    virtual void release_variables() override;
};

class FiniteDiffPathtracingIntegrator : public Integrator
{
public:
    FiniteDiffPathtracingIntegrator(const atcg::ref_ptr<RaytracingContext>& context, const Dictionary& dict);

    ~FiniteDiffPathtracingIntegrator();

    virtual void onImGuiRender() override;

    virtual void generateRays(Dictionary& in_out_dictionary) override;

    virtual void reset() override;

private:
    friend class FiniteDiffPathNode;

    atcg::ref_ptr<Integrator> _integrator;
};
}    // namespace atcg