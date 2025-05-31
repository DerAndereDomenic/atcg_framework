#pragma once

#include <Integrator/Integrator.h>
#include <Shape/IAS.h>
#include <Integrator/PathtracingData.cuh>
#include <Emitter/EnvironmentEmitter.h>
#include <Emitter/PointEmitter.h>

namespace atcg
{
/**
 * @brief A simple path tracer
 */
class PathtracingIntegrator : public Integrator
{
public:
    /**
     * @brief Constructor
     *
     * @param context The raytracing context
     */
    PathtracingIntegrator(const atcg::ref_ptr<RaytracingContext>& context);

    /**
     * @brief Destructor
     */
    ~PathtracingIntegrator();

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
     *
     * @param camera The camera
     * @param output The vector with output tensors
     */
    virtual void generateRays(const atcg::ref_ptr<PerspectiveCamera>& camera,
                              const std::vector<torch::Tensor>& output) override;

    /**
     * @brief Reset the internal structure of the integrator
     */
    virtual void reset() override;

private:
    uint32_t _raygen_index;
    uint32_t _surface_miss_index;
    uint32_t _occlusion_miss_index;

    std::vector<atcg::ref_ptr<ShapeInstance>> _shapes;

    atcg::DeviceBuffer<const EmitterVPtrTable*> _emitters;
    atcg::ref_ptr<EnvironmentEmitter> _environment_emitter = nullptr;
    std::vector<atcg::ref_ptr<PointEmitter>> _emitter;

    atcg::ref_ptr<InstanceAccelerationStructure> _ias;
    atcg::dref_ptr<PathtracingParams> _launch_params;
    uint32_t _frame_counter = 0;

    torch::Tensor _accumulation_buffer;
};
}    // namespace atcg