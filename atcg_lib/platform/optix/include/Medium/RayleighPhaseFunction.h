#pragma once

#include <Medium/PhaseFunction.h>

namespace atcg
{
class RayleighPhaseFunction : public PhaseFunction
{
public:
    RayleighPhaseFunction(const atcg::Dictionary& dict);

    virtual ~RayleighPhaseFunction();

    virtual void onImGuiRender() override {}

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
};
}    // namespace atcg