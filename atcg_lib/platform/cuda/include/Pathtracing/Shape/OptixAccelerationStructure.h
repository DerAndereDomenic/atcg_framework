#pragma once

#include <Pathtracing/AccelerationStructure.h>
#include <Pathtracing/OptixInterface.h>
#include <optix.h>

#include <Scene/Scene.h>

namespace atcg
{

/**
 * @brief Interface to create an acceleration structure
 */
class OptixAccelerationStructure : public AccelerationStructure, public OptixComponent
{
public:
    /**
     * @brief Initialize the optix pipeline.
     * Each Optix component has to initialize its part of the raytracing pipeline by defining appropriate entry points
     * and sbt entries.
     *
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override = 0;

    /**
     * @brief Get the traversable handle of the AST
     *
     * @return The handle
     */
    OptixTraversableHandle getTraversableHandle() const { return _handle; }

    /**
     * @brief Get the hitgroup program
     *
     * @return The hit group
     */
    OptixProgramGroup getHitGroup() const { return _hit_group; }

protected:
    atcg::DeviceBuffer<uint8_t> _ast_buffer;
    OptixTraversableHandle _handle;
    OptixProgramGroup _hit_group;
};

/**
 * @brief Acceleration structure for single mesh objects
 */
class MeshAccelerationStructure : public OptixAccelerationStructure
{
public:
    /**
     * @brief Create a mesh acceleration structure
     *
     * @param context The optix context
     * @param graph The geometry to build the AST for
     */
    MeshAccelerationStructure(OptixDeviceContext context, const atcg::ref_ptr<Graph>& graph);

    /**
     * @brief Destructor
     */
    virtual ~MeshAccelerationStructure();

    /**
     * @brief Initialize the optix pipeline.
     * Each Optix component has to initialize its part of the raytracing pipeline by defining appropriate entry points
     * and sbt entries.
     *
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
};

/**
 * @brief Class to model a AST for multiple instances of the same geometry
 */
class IASAccelerationStructure : public OptixAccelerationStructure
{
public:
    /**
     * @brief Create the AST
     *
     * @param context The optix context
     * @param scene The scene to build the IAS for
     */
    IASAccelerationStructure(OptixDeviceContext context, const atcg::ref_ptr<Scene>& scene);

    /**
     * @brief Destructor
     */
    ~IASAccelerationStructure();

    /**
     * @brief Initialize the optix pipeline.
     * Each Optix component has to initialize its part of the raytracing pipeline by defining appropriate entry points
     * and sbt entries.
     *
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    atcg::ref_ptr<Scene> _scene;
};
}    // namespace atcg