#pragma once

#include <Integrator/Integrator.h>
#include <Integrator/IntegratorRegistry.h>
#include <Shape/IAS.h>
#include <Integrator/PathtracingData.cuh>
#include <Emitter/EnvironmentEmitter.h>
#include <Emitter/PointEmitter.h>
#include <Scene/OptixScene.h>

namespace atcg
{
/**
 * @brief A simple path tracer
 */
class ATCG_API PathtracingIntegrator : public Integrator
{
public:
    /**
     * @brief Constructor
     *
     * @param context The raytracing context
     * @param dict Additional parameters
     */
    PathtracingIntegrator(const atcg::ref_ptr<RaytracingContext>& context, const atcg::Dictionary& dict);

    /**
     * @brief Destructor
     */
    virtual ~PathtracingIntegrator();

    /**
     * @brief Generate the rays and write to some output tensors
     * This integrator expects:
     * camera - atcg::ref_ptr<PerspectiveCamera>
     * output - torch::Tensor
     *
     * @param in_out_dictionary The input output dictionary
     */
    virtual void generateRays(Dictionary& in_out_dictionary) override;

    /**
     * @brief Reset the internal structure of the integrator
     */
    virtual void reset() override;

    static void registerIntegrator(IntegratorRegistry::Registry* registry);

private:
    /**
     * @brief Initialize a pipeline.
     */
    void initializePipeline(const Dictionary& dict);

    uint32_t _raygen_index;
    uint32_t _surface_miss_index;
    uint32_t _occlusion_miss_index;

    atcg::ref_ptr<Scene> _scene;
    atcg::ref_ptr<OptixScene> _optix_scene;
    atcg::dref_ptr<PathtracingParams> _launch_params;
    uint32_t _frame_counter = 0;
};
}    // namespace atcg