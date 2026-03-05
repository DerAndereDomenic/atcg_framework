#pragma once

#include <Integrator/Integrator.h>
#include <Shape/IAS.h>
#include "VolDiffPathtracingData.cuh"
#include <Emitter/EnvironmentEmitter.h>
#include <Emitter/PointEmitter.h>
#include <Scene/OptixScene.h>
#include <Scene/SceneHierarchyPanel.h>
#include "DifferentiableIntegrator.h"

#include <torch/torch.h>

namespace atcg
{
class VolDiffPathtracingIntegrator;

struct VolDiffPathNode : public torch::autograd::Node
{
    VolDiffPathtracingIntegrator* integrator;
    torch::Tensor sample;
    uint32_t rng_index;
    PerspectiveCamera* camera;
    torch::autograd::variable_list apply(torch::autograd::variable_list&& grads) override;

    virtual void release_variables() override;
};

/**
 * @brief A simple path tracer
 */
class VolDiffPathtracingIntegrator : public DifferentiableIntegrator
{
public:
    /**
     * @brief Constructor
     *
     * @param context The raytracing context
     * @param dict Additional parameters
     */
    VolDiffPathtracingIntegrator(const atcg::ref_ptr<RaytracingContext>& context, const atcg::Dictionary& dict);
    /**
     * @brief Destructor
     */
    ~VolDiffPathtracingIntegrator();

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() override;

    virtual torch::Tensor sample(Dictionary& in_out_dictionary) override;

    /**
     * @brief Reset the internal structure of the integrator
     */
    virtual void reset() override;

    virtual std::vector<torch::Tensor> getParameters() const override;

    virtual std::vector<torch::Tensor> getParameterGradients() const override;

    virtual void clampParameters() override;

    virtual void markOptimizable() override;

    virtual void zeroGrad() override;

private:
    friend class VolDiffPathNode;
    /**
     * @brief Initialize a pipeline.
     * This function should be overwritten by each child class and it should add its functions to the pipeline and the
     * sbt.
     *
     * @param pipeline The pipeline
     * @param sbt The shader binding table
     */
    void initializePipeline(const Dictionary& dict);


    torch::Tensor _forwardTrace(Dictionary& in_out_dictionary);
    void _backwardTrace(Dictionary& in_out_dictionary);

private:
    uint32_t _raygen_index_forward;
    uint32_t _raygen_index_backward;
    uint32_t _surface_miss_index;
    uint32_t _occlusion_miss_index;

    atcg::ref_ptr<OptixScene> _optix_scene;
    atcg::dref_ptr<VolDiffPathtracingParams> _launch_params;

    std::vector<Differentiable*> _differentiable_components;

    GUI::SceneHierarchyPanel _panel = GUI::SceneHierarchyPanel("DiffPath");
};
}    // namespace atcg