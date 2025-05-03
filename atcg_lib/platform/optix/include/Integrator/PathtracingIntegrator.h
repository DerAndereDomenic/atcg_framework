#pragma once

#include <Integrator/Integrator.h>
#include <Shape/IAS.h>
#include <Integrator/PathtracingData.cuh>
#include <Emitter/EnvironmentEmitter.h>
#include <Emitter/PointEmitter.h>

namespace atcg
{
class PathtracingIntegrator : public Integrator
{
public:
    PathtracingIntegrator(const atcg::ref_ptr<RaytracingContext>& context);

    ~PathtracingIntegrator();

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

    virtual void generateRays(const atcg::ref_ptr<PerspectiveCamera>& camera,
                              const std::vector<torch::Tensor>& output) override;

    virtual void reset() override;

private:
    uint32_t _raygen_index;
    uint32_t _surface_miss_index;
    uint32_t _occlusion_miss_index;

    std::vector<atcg::ref_ptr<ShapeInstance>> _shapes;

    atcg::DeviceBuffer<const EmitterVPtrTable*> _emitters;
    atcg::ref_ptr<EnvironmentEmitter> _environment_emitter = nullptr;
    std::vector<atcg::ref_ptr<PointEmitter>> _emitter;

    atcg::ref_ptr<IAS> _ias;
    atcg::dref_ptr<PathtracingParams> _launch_params;
    uint32_t _frame_counter = 0;

    torch::Tensor _accumulation_buffer;
};
}    // namespace atcg