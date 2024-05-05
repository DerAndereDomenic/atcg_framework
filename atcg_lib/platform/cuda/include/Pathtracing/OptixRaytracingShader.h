#pragma once

#include <Pathtracing/OptixInterface.h>
#include <Pathtracing/RaytracingShader.h>

namespace atcg
{
class OptixRaytracingShader : public OptixComponent, public RaytracingShader
{
public:
    OptixRaytracingShader(OptixDeviceContext context) : _context(context) {}

    virtual ~OptixRaytracingShader() {}

    virtual void reset() = 0;

    virtual void generateRays(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                              const atcg::ref_ptr<ShaderBindingTable>& sbt,
                              torch::Tensor& output) = 0;

protected:
    OptixDeviceContext _context;
};
}    // namespace atcg