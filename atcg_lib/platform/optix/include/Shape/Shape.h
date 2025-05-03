#pragma once

#include <Core/Memory.h>
#include <Core/OptixComponent.h>
#include <Core/RaytracingContext.h>
#include <Shape/ShapeData.cuh>

#include <optix.h>

namespace atcg
{
class ShapeInstance;
class Shape : public OptixComponent
{
public:
    virtual ~Shape() {}

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) = 0;

    virtual void prepareAccelerationStructure(const atcg::ref_ptr<RaytracingContext>& context) = 0;

    ATCG_INLINE OptixTraversableHandle getAST() { return _ast_handle; }

    ATCG_INLINE OptixProgramGroup getHitGroup() const { return _hit_group; }

protected:
    friend class ShapeInstance;
    atcg::DeviceBuffer<uint8_t> _ast_buffer;
    OptixTraversableHandle _ast_handle = 0;
    OptixProgramGroup _hit_group;

    ShapeData* _shape_data;
};
}    // namespace atcg