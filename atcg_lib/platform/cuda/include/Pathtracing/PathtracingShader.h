#pragma once

#include <Pathtracing/OptixInterface.h>
#include <Pathtracing/Params.cuh>

namespace atcg
{
class PathtracingShader : public OptixRaytracingShader
{
public:
    PathtracingShader(OptixDeviceContext context,
                      const atcg::ref_ptr<Scene>& scene,
                      const atcg::ref_ptr<PerspectiveCamera>& camera);

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
    atcg::DeviceBuffer<uint8_t> _ias_buffer;
    OptixTraversableHandle _ias_handle;

    glm::mat4 _inv_camera_view;
    float _fov_y;

    torch::Tensor _accumulation_buffer;

    cudaStream_t _stream;

    atcg::ref_ptr<OptixEmitter> _environment_emitter = nullptr;
    atcg::DeviceBuffer<const atcg::EmitterVPtrTable*> _emitter_tables;

    uint32_t _frame_counter = 0;
    uint32_t _raygen_index  = 0;
};
}    // namespace atcg