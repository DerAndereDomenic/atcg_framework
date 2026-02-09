#pragma once

#include <Integrator/Integrator.h>
#include <Core/OptixComponent.h>

namespace atcg
{
class DifferentiableIntegrator : public Integrator, public Differentiable
{
public:
    /**
     * @brief Constructor
     *
     * @param context The raytracing context
     * @param dict Additional parameters
     */
    DifferentiableIntegrator(const atcg::ref_ptr<RaytracingContext>& context, const atcg::Dictionary& dict)
        : Integrator(context, dict)
    {
    }

    // Differentiable
    virtual torch::Tensor sample(Dictionary& in_out_dictionary) = 0;

    virtual void generateRays(Dictionary& in_out_dictionary) override
    {
        auto result = sample(in_out_dictionary);
        in_out_dictionary.setValue("output_img", result);
    }
};
}    // namespace atcg