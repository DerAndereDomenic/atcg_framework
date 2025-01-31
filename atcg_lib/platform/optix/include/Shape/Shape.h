#pragma once

#include <Core/Memory.h>
#include <Core/OptixComponent.h>

#include <optix.h>

namespace atcg
{
class Shape : public OptixComponent
{
public:
    virtual ~Shape() {}

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) = 0;

    virtual void prepareAccelerationStructure(OptixDeviceContext context) = 0;

    ATCG_INLINE OptixTraversableHandle getAST() { return _ast_handle; }

    ATCG_INLINE OptixProgramGroup getHitGroup() const { return _hit_group; }

protected:
    atcg::DeviceBuffer<uint8_t> _ast_buffer;
    OptixTraversableHandle _ast_handle = 0;
    OptixProgramGroup _hit_group;
};
}    // namespace atcg