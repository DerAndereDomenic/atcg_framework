#pragma once

#include <Core/API.h>
#include <Core/Memory.h>
#include <Core/RaytracingComponent.h>
#include <DataStructure/Dictionary.h>
#include <Renderer/RaytracingContext.h>
#include <Shape/ShapeData.cuh>

#ifndef __CUDACC__
    #include <Scene/ComponentGUIHandler.h>
#endif

#include <Core/Optix.h>

namespace atcg
{
class ShapeInstance;
/**
 * @brief Class to model a shape
 */
class ATCG_API Shape : public RaytracingComponent
{
public:
    /**
     * @brief Constructor
     */
    Shape() = default;

    /**
     * @brief Create a shape with parameters
     *
     * @param dict The parameters
     */
    Shape(const atcg::Dictionary& dict) {}

    /**
     * @brief Destructor
     */
    virtual ~Shape() {}

    /**
     * @brief Prepare the acceleration structure of the shape
     *
     * @param context The raytracing context
     */
    virtual void prepareAccelerationStructure(const atcg::ref_ptr<RaytracingContext>& context) = 0;

    /**
     * @brief Get the shape type
     *
     * @return The shape type
     */
    virtual std::string getShapeType() const = 0;

    /**
     * @brief Get the AST handle
     *
     * @return The handle
     */
    ATCG_INLINE OptixTraversableHandle getAST() { return _ast_handle; }

    ATCG_INLINE ShapeData* getShapeData() const { return _shape_data; }

protected:
    friend class ShapeInstance;
    atcg::DeviceBuffer<uint8_t> _ast_buffer;
    OptixTraversableHandle _ast_handle = 0;
    std::vector<OptixProgramGroup> _hit_groups;

    ShapeData* _shape_data;
};
}    // namespace atcg