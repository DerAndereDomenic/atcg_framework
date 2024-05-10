#pragma once

#include <Pathtracing/OptixRaytracingShader.h>
#include <Pathtracing/Shader/PathtracingShader.cuh>
#include <Pathtracing/OptixAccelerationStructure.h>

namespace atcg
{
class PathtracingShader : public OptixRaytracingShader
{
public:
    PathtracingShader(OptixDeviceContext context);

    ~PathtracingShader();

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

    virtual void reset() override;

    virtual void setCamera(const atcg::ref_ptr<PerspectiveCamera>& camera) override;

    virtual void generateRays(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                              const atcg::ref_ptr<ShaderBindingTable>& sbt,
                              torch::Tensor& output) override;

private:
    atcg::dref_ptr<Params> _launch_params;

    glm::mat4 _inv_camera_view;
    float _fov_y;

    cudaStream_t _stream;

    atcg::ref_ptr<OptixEmitter> _environment_emitter = nullptr;
    atcg::DeviceBuffer<const atcg::EmitterVPtrTable*> _emitter_tables;

    uint32_t _frame_counter = 0;
    uint32_t _raygen_index  = 0;

    atcg::ref_ptr<IASAccelerationStructure> _accel;
};
}    // namespace atcg