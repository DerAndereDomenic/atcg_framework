#pragma once

#include <Integrator/Integrator.h>
#include <Integrator/IntegratorRegistry.h>
#include <Shape/IAS.h>
#include <Integrator/PhotonMapData.cuh>
#include <Emitter/EnvironmentEmitter.h>
#include <Emitter/PointEmitter.h>
#include <Scene/OptixScene.h>
#include <cuBQL/bvh.h>

namespace atcg
{
/**
 * @brief A simple photon mapping integrator
 */
class ATCG_API PhotonMapIntegrator : public Integrator
{
public:
    /**
     * @brief Constructor
     *
     * @param context The raytracing context
     * @param dict Additional parameters
     */
    PhotonMapIntegrator(const atcg::ref_ptr<RaytracingContext>& context, const atcg::Dictionary& dict);

    /**
     * @brief Destructor
     */
    virtual ~PhotonMapIntegrator();

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

    void generateNewPhotonMap();

    void tracePhotons();

    void buildPhotonMap();

    uint32_t _raygen_index;
    uint32_t _raygen_sample_photons_index;
    uint32_t _raygen_photon_request_index;
    uint32_t _surface_miss_index;
    uint32_t _occlusion_miss_index;

    atcg::ref_ptr<Scene> _scene;
    atcg::ref_ptr<OptixScene> _optix_scene;
    atcg::dref_ptr<PhotonMapParams> _launch_params;
    uint32_t _frame_counter = 0;

    atcg::DeviceBuffer<PhotonMapData> _photon_data;
    atcg::DeviceBuffer<cuBQL::box3f> _photon_bounds;
    atcg::DeviceBuffer<PhotonGatherData> _photon_gather_data;
    atcg::dref_ptr<int> _photon_index;
    atcg::dref_ptr<cuBQL::bvh3f> _device_photon_bvh;
    cuBQL::bvh3f _photon_bvh;
};
}    // namespace atcg