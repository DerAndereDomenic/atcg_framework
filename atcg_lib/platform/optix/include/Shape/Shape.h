#pragma once

#include <Core/Memory.h>
#include <Core/OptixComponent.h>
#include <Core/RaytracingContext.h>
#include <Shape/ShapeData.cuh>

#include <optix.h>

namespace atcg
{
class ShapeInstance;
/**
 * @brief Class to model a shape
 */
class Shape : public OptixComponent
{
public:
    /**
     * @brief Destructor
     */
    virtual ~Shape() {}

    /**
     * @brief Initialize a pipeline.
     * This function should be overwritten by each child class and it should add its functions to the pipeline and the
     * sbt.
     *
     * @param pipeline The pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) = 0;

    /**
     * @brief Prepare the acceleration structure of the shape
     * 
     * @param context The raytracing context
     */
    virtual void prepareAccelerationStructure(const atcg::ref_ptr<RaytracingContext>& context) = 0;

    /**
     * @brief Get the AST handle
     * 
     * @return The handle
     */
    ATCG_INLINE OptixTraversableHandle getAST() { return _ast_handle; }

    /**
     * @brief Get the hit group
     * 
     * @return The hit group
     */
    ATCG_INLINE OptixProgramGroup getHitGroup() const { return _hit_group; }

protected:
    friend class ShapeInstance;
    atcg::DeviceBuffer<uint8_t> _ast_buffer;
    OptixTraversableHandle _ast_handle = 0;
    OptixProgramGroup _hit_group;

    ShapeData* _shape_data;
};
}    // namespace atcg