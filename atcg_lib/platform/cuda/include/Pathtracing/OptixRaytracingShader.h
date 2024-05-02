#pragma once

#include <Pathtracing/OptixInterface.h>
#include <Scene/Scene.h>

namespace atcg
{
class OptixRaytracingShader : public OptixComponent, public RaytracingShader
{
public:
    OptixRaytracingShader(OptixDeviceContext context) : _context(context) {}

    virtual ~OptixRaytracingShader() {}

    virtual void reset() = 0;

    virtual void setScene(const atcg::ref_ptr<Scene>& scene) { _scene = scene; }

    virtual void setCamera(const atcg::ref_ptr<PerspectiveCamera>& camera) { _camera = camera; };

    virtual void generateRays(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                              const atcg::ref_ptr<ShaderBindingTable>& sbt,
                              torch::Tensor& output) = 0;

protected:
    OptixDeviceContext _context;
    atcg::ref_ptr<Scene> _scene;
    atcg::ref_ptr<PerspectiveCamera> _camera;
};
}    // namespace atcg