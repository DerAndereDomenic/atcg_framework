#pragma once

#include <Integrator/Integrator.h>
#include <Shape/IAS.h>
#include "AttachedDiffPathtracingData.cuh"
#include <Emitter/EnvironmentEmitter.h>
#include <Emitter/PointEmitter.h>
#include <Scene/OptixScene.h>
#include <Scene/SceneHierarchyPanel.h>

#include <torch/torch.h>

namespace atcg
{
class AttachedDiffPathtracingIntegrator;

struct AttachedDiffPathNode : public torch::autograd::Node
{
    AttachedDiffPathtracingIntegrator* integrator;
    torch::Tensor sample;
    torch::Tensor JL;
    uint32_t rng_index;
    torch::autograd::variable_list apply(torch::autograd::variable_list&& grads) override;

    virtual void release_variables() override;
};

/**
 * @brief A simple path tracer
 */
class AttachedDiffPathtracingIntegrator : public Integrator
{
public:
    /**
     * @brief Constructor
     *
     * @param context The raytracing context
     * @param dict Additional parameters
     */
    AttachedDiffPathtracingIntegrator(const atcg::ref_ptr<RaytracingContext>& context, const atcg::Dictionary& dict);

    /**
     * @brief Destructor
     */
    ~AttachedDiffPathtracingIntegrator();

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() override;

    /**
     * @brief Reset the internal structure of the integrator
     */
    virtual void reset() override;

    virtual void generateRays(Dictionary& in_out_dictionary) override;

    std::tuple<torch::Tensor, torch::Tensor> forwardTrace(Dictionary& in_out_dictionary);
    void backwardTrace(Dictionary& in_out_dictionary);

    ATCG_INLINE torch::Tensor getAOVBuffer(uint32_t index) const
    {
        if(index >= _aov_buffers.size())
        {
            throw std::out_of_range("AOV buffer index out of range");
        }
        return _aov_buffers[index];
    }

private:
    friend class AttachedDiffPathNode;

    /**
     * @brief Initialize a pipeline.
     * This function should be overwritten by each child class and it should add its functions to the pipeline and the
     * sbt.
     *
     * @param pipeline The pipeline
     * @param sbt The shader binding table
     */
    void initializePipeline(const Dictionary& dict);


private:
    uint32_t _raygen_index_forward;
    uint32_t _surface_miss_index;
    uint32_t _dual_miss_index;
    uint32_t _occlusion_miss_index;

    atcg::ref_ptr<OptixScene> _optix_scene;
    atcg::dref_ptr<AttachedDiffPathtracingParams> _launch_params;

    GUI::SceneHierarchyPanel _panel = GUI::SceneHierarchyPanel("DiffPath");

    uint32_t _frame_counter = 0;

    std::vector<torch::Tensor> _aov_buffers;
    atcg::DeviceBuffer<float*> _aov_buffer_pointers;
};

}    // namespace atcg