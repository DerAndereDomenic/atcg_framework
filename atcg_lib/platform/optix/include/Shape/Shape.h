#pragma once

#include <Core/Memory.h>
#include <Core/OptixComponent.h>
#include <DataStructure/Dictionary.h>
#include <Core/RaytracingContext.h>
#include <Shape/ShapeData.cuh>

#ifndef __CUDACC__
    #include <Scene/ComponentGUIHandler.h>
#endif

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
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() = 0;

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

    ATCG_INLINE ShapeData* getShapeData() const { return _shape_data; }

protected:
    friend class ShapeInstance;
    atcg::DeviceBuffer<uint8_t> _ast_buffer;
    OptixTraversableHandle _ast_handle = 0;
    OptixProgramGroup _hit_group;

    ShapeData* _shape_data;
};

struct ShapeComponent
{
    ShapeComponent() = default;
    ShapeComponent(const atcg::ref_ptr<Shape>& shape) : shape(shape) {}

    atcg::ref_ptr<Shape> shape;

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "ShapeComponent"; }
};

#ifndef __CUDACC__
namespace GUI
{
ATCG_DECLARE_COMPONENT_GUI_RENDERER(ShapeComponent);
}    // namespace GUI
#endif
}    // namespace atcg