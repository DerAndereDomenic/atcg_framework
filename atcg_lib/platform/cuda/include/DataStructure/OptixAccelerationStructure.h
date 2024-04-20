#pragma once

#include <DataStructure/AccelerationStructure.h>
#include <Renderer/OptixInterface.h>
#include <optix.h>

namespace atcg
{
class OptixAccelerationStructure : public AccelerationStructure, public OptixComponent
{
public:
    OptixAccelerationStructure(OptixDeviceContext context, const atcg::ref_ptr<Graph>& graph);

    ~OptixAccelerationStructure();

    OptixTraversableHandle getTraversableHandle() const { return _handle; }

    OptixProgramGroup getHitGroup() const { return _hit_group; }

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    atcg::DeviceBuffer<uint8_t> _gas_buffer;
    OptixTraversableHandle _handle;
    OptixProgramGroup _hit_group;
};
}    // namespace atcg