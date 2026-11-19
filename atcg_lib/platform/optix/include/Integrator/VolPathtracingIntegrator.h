#pragma once

#include <Integrator/Integrator.h>
#include <Integrator/IntegratorRegistry.h>
#include <Shape/IAS.h>
#include <Integrator/VolPathtracingData.cuh>
#include <Emitter/EnvironmentEmitter.h>
#include <Emitter/PointEmitter.h>
#include <Sensor/Sensor.h>
#include <Scene/OptixScene.h>
#include <GUI/SceneHierarchyPanel.h>
#include <Integrator/IntegratorSettings.h>

namespace atcg
{
/**
 * @brief A simple path tracer
 */
class ATCG_API VolPathtracingIntegrator : public Integrator
{
public:
    /**
     * @brief Constructor
     *
     * @param context The raytracing context
     * @param dict Additional parameters
     */
    VolPathtracingIntegrator(const atcg::ref_ptr<RaytracingContext>& context, const atcg::Dictionary& dict);

    /**
     * @brief Destructor
     */
    virtual ~VolPathtracingIntegrator();

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

    static void registerIntegrator(IntegratorRegistry::Registry* registry);

    static std::string getType() { return "VolPathtracing"; }

private:
    /**
     * @brief Initialize a pipeline.
     */
    virtual void initializePipeline(const Dictionary& dict);

    uint32_t _raygen_index;
    uint32_t _surface_miss_index;

    atcg::ref_ptr<OptixScene> _optix_scene;
    atcg::dref_ptr<VolPathtracingParams> _launch_params;
    uint32_t _frame_counter = 0;

    GUI::SceneHierarchyPanel _panel = GUI::SceneHierarchyPanel("VolPath");

    // Settings:
    bool _use_rr        = true;
    uint32_t _max_depth = 512;
    bool _hide_emitters = false;
};

template<>
struct IntegratorSettings<VolPathtracingIntegrator> : public IntegratorSettingsBase
{
    std::string getType() const override { return VolPathtracingIntegrator::getType(); }

    bool renderSettings() override;

    atcg::Dictionary toDictionary() const override;

    nlohmann::json toJson() const override;

    void fromJson(const nlohmann::json& j) override;

    bool _use_rr        = true;
    uint32_t _max_depth = 512;
    bool _hide_emitters = false;
};

}    // namespace atcg