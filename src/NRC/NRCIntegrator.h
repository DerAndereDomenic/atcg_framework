#pragma once

#include <Integrator/Integrator.h>
#include <Shape/IAS.h>
#include "NRCIntegratorData.cuh"
#include <Emitter/EnvironmentEmitter.h>
#include <Emitter/PointEmitter.h>
#include <Scene/OptixScene.h>
#include <Neural/MLP.h>
#include <Neural/HashGrid.h>
#include <torch/torch.h>
#include <DataStructure/Statistics.h>

namespace atcg
{

using NRCMLP      = MLP<NRC_NUM_HIDDEN_LAYERS, NRC_INPUT_SIZE, NRC_HIDDEN_LAYER_SIZE, NRC_OUTPUT_SIZE>;
using NRCHashGrid = HashGrid<half, NRC_HASH_GRID_LEVELS, NRC_HASH_GRID_FEATURES_PER_LEVEL>;

/**
 * @brief A simple path tracer
 */
class NRCIntegrator : public Integrator
{
public:
    /**
     * @brief Constructor
     *
     * @param context The raytracing context
     * @param dict Additional parameters
     */
    NRCIntegrator(const atcg::ref_ptr<RaytracingContext>& context, const atcg::Dictionary& dict);

    /**
     * @brief Destructor
     */
    virtual ~NRCIntegrator();


    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() override;

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

private:
    /**
     * @brief Initialize a pipeline.
     */
    void initializePipeline(const Dictionary& dict);

    void generateTrainingSamples();

    void trainRadianceCache();

    void renderWithRadianceCache(Dictionary& in_out_dictionary);

private:
    uint32_t _raygen_render;
    uint32_t _raygen_sample_gen;
    uint32_t _raygen_train;
    uint32_t _surface_miss_index;
    uint32_t _occlusion_miss_index;

    atcg::ref_ptr<Scene> _scene;
    atcg::ref_ptr<OptixScene> _optix_scene;
    atcg::dref_ptr<atcg::BoundingBox> _scene_aabb;
    atcg::dref_ptr<NRCParams> _launch_params;
    uint32_t _frame_counter = 0;

    uint32_t _max_num_training_samples;
    atcg::DeviceBuffer<TrainingSample> _training_samples;
    atcg::DeviceBuffer<SampledSpectrum> _training_sample_radiance;
    atcg::dref_ptr<int> _training_samples_queue_index;

    // Radiance cache
    NRCMLP _mlp;
    NRCHashGrid _hash_grid;
    torch::Tensor _weights, _bias, _hash_weights;
    atcg::ref_ptr<torch::optim::Adam> _optimizer;

    // Some statistics
    atcg::Statistic<float> _sample_generation_time;
    atcg::Statistic<float> _training_time;
    atcg::Statistic<float> _render_time;
};
}    // namespace atcg