#pragma once

#include <Scene/Components.h>
#include <Emitter/Emitter.h>
#include <Emitter/PointEmitterData.cuh>

namespace atcg
{
class PointEmitter : public Emitter
{
public:
    PointEmitter(const glm::vec3& position, const PointLightComponent& point_light);

    ~PointEmitter();

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    atcg::dref_ptr<PointEmitterData> _point_emitter_data;
};
}    // namespace atcg