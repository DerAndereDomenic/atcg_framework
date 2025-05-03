#pragma once

#include <optix.h>

#include <Core/Platform.h>
#include <Core/RaytracingPipeline.h>
#include <Core/ShaderBindingTable.h>
#include <Core/OptixComponent.h>
#include <Core/RaytracingContext.h>
#include <Scene/Scene.h>
#include <Renderer/PerspectiveCamera.h>

namespace atcg
{
class Integrator : public OptixComponent
{
public:
    Integrator(const atcg::ref_ptr<RaytracingContext>& context) : _context(context) {}

    virtual ~Integrator() {}

    ATCG_INLINE void setScene(const atcg::ref_ptr<Scene>& scene) { _scene = scene; }

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) = 0;

    virtual void generateRays(const atcg::ref_ptr<PerspectiveCamera>& camera,
                              const std::vector<torch::Tensor>& output) = 0;

    virtual void reset() = 0;

protected:
    atcg::ref_ptr<RaytracingContext> _context;
    atcg::ref_ptr<Scene> _scene;

    atcg::ref_ptr<RayTracingPipeline> _pipeline;
    atcg::ref_ptr<ShaderBindingTable> _sbt;
};
}    // namespace atcg