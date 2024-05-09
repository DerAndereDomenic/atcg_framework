#pragma once

#include <Pathtracing/AccelerationStructure.h>
#include <Pathtracing/OptixInterface.h>
#include <optix.h>

#include <Scene/Scene.h>

namespace atcg
{
class GASAccelerationStructure : public AccelerationStructure, public OptixComponent
{
public:
    GASAccelerationStructure(OptixDeviceContext context, const atcg::ref_ptr<Graph>& graph);

    ~GASAccelerationStructure();

    OptixTraversableHandle getTraversableHandle() const { return _handle; }

    OptixProgramGroup getHitGroup() const { return _hit_group; }

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    atcg::DeviceBuffer<uint8_t> _gas_buffer;
    OptixTraversableHandle _handle;
    OptixProgramGroup _hit_group;
};

class IASAccelerationStructure : public AccelerationStructure, public OptixComponent
{
public:
    IASAccelerationStructure(OptixDeviceContext context, const atcg::ref_ptr<Scene>& scene);

    ~IASAccelerationStructure();

    OptixTraversableHandle getTraversableHandle() const { return _handle; }

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    atcg::DeviceBuffer<uint8_t> _ias_buffer;
    OptixTraversableHandle _handle;
    atcg::ref_ptr<Scene> _scene;
};
}    // namespace atcg