#pragma once

#include <optix.h>

#include <Core/Platform.h>
#include <Core/RaytracingPipeline.h>
#include <Core/ShaderBindingTable.h>
#include <Core/OptixComponent.h>
#include <Core/RaytracingContext.h>
#include <Scene/Scene.h>
#include <Renderer/PerspectiveCamera.h>
#include <DataStructure/TorchUtils.h>

#include <vector>

namespace atcg
{
/**
 * @brief A class to model an integrator
 */
class Integrator : public OptixComponent
{
public:
    /**
     * @brief Constructor
     *
     * @param context The raytracing context
     */
    Integrator(const atcg::ref_ptr<RaytracingContext>& context) : _context(context) {}

    /**
     * @brief Destructor
     */
    virtual ~Integrator() {}

    /**
     * @brief Set the scene
     *
     * @param scene The scene
     */
    ATCG_INLINE void setScene(const atcg::ref_ptr<Scene>& scene) { _scene = scene; }

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
     * @brief Generate the rays and write to some output tensors
     *
     * @param camera The camera
     * @param output The vector with output tensors
     */
    virtual void generateRays(const atcg::ref_ptr<PerspectiveCamera>& camera,
                              const std::vector<torch::Tensor>& output) = 0;

    /**
     * @brief Reset the internal structure of the integrator
     */
    virtual void reset() = 0;

protected:
    atcg::ref_ptr<RaytracingContext> _context;
    atcg::ref_ptr<Scene> _scene;

    atcg::ref_ptr<RayTracingPipeline> _pipeline;
    atcg::ref_ptr<ShaderBindingTable> _sbt;
};
}    // namespace atcg