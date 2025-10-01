#pragma once

#include <Integrator/Integrator.h>
#include <Shape/IAS.h>
#include "DiffPathtracingData.cuh"
#include <Emitter/EnvironmentEmitter.h>
#include <Emitter/PointEmitter.h>
#include <Scene/OptixScene.h>

namespace atcg
{
/**
 * @brief A simple path tracer
 */
class DiffPathtracingIntegrator : public Integrator
{
public:
    /**
     * @brief Constructor
     *
     * @param context The raytracing context
     * @param dict Additional parameters
     */
    DiffPathtracingIntegrator(const atcg::ref_ptr<RaytracingContext>& context, const atcg::Dictionary& dict);

    /**
     * @brief Destructor
     */
    ~DiffPathtracingIntegrator();

    /**
     * @brief Initialize a pipeline.
     * This function should be overwritten by each child class and it should add its functions to the pipeline and the
     * sbt.
     *
     * @param pipeline The pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

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

    torch::Tensor getHDR() const;

    void toggleOptimization();

private:
    void forwardPass(Dictionary& in_out_dictionary);
    void backwardPass(Dictionary& in_out_dictionary);

private:
    uint32_t _raygen_index_forward;
    uint32_t _raygen_index_backward;
    uint32_t _surface_miss_index;
    uint32_t _occlusion_miss_index;

    atcg::ref_ptr<OptixScene> _optix_scene;
    atcg::dref_ptr<DiffPathtracingParams> _launch_params;
    uint32_t _frame_counter = 0;

    torch::Tensor _accumulation_buffer;
    torch::Tensor _adjoint_x;
    torch::Tensor _adjoint_y;
    bool _optimize = false;

    std::vector<Differentiable*> _differentiable_components;
};
}    // namespace atcg