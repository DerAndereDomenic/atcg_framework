#pragma once

#include <optix.h>

#include <Core/Platform.h>
#include <Core/RaytracingPipeline.h>
#include <Core/ShaderBindingTable.h>
#include <Core/OptixComponent.h>
#include <Scene/Scene.h>
#include <Renderer/PerspectiveCamera.h>

namespace atcg
{
class Integrator : public OptixComponent
{
public:
    Integrator(OptixDeviceContext context) : _context(context) {}

    virtual ~Integrator() {}

    ATCG_INLINE void setScene(const atcg::ref_ptr<Scene>& scene) { _scene = scene; }

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) = 0;

    virtual void generateRays(const atcg::ref_ptr<PerspectiveCamera>& camera, torch::Tensor& output) = 0;

protected:
    OptixDeviceContext _context;
    atcg::ref_ptr<Scene> _scene;

    atcg::ref_ptr<RayTracingPipeline> _pipeline;
    atcg::ref_ptr<ShaderBindingTable> _sbt;
};
}    // namespace atcg