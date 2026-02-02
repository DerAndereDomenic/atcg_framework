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

    virtual torch::Tensor getHDR() const = 0;
};
}    // namespace atcg