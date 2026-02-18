#pragma once

#include "DiffPathtracingIntegrator.h"
#include "DifferentiableIntegrator.h"

#include <torch/torch.h>

namespace atcg
{
struct FiniteDiffPathNode : public torch::autograd::Node
{
    DiffPathtracingIntegrator* integrator;
    uint32_t rng_index;
    uint32_t width, height;
    atcg::ref_ptr<PerspectiveCamera> camera;
    torch::autograd::variable_list apply(torch::autograd::variable_list&& grads) override;

    virtual void release_variables() override;
};

class FiniteDiffPathtracingIntegrator : public DiffPathtracingIntegrator
{
public:
    FiniteDiffPathtracingIntegrator(const atcg::ref_ptr<RaytracingContext>& context, const Dictionary& dict);

    ~FiniteDiffPathtracingIntegrator();

    virtual torch::Tensor sample(Dictionary& in_out_dictionary) override;

private:
    friend class FiniteDiffPathNode;
};
}    // namespace atcg