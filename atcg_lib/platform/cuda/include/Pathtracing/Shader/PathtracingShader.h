#pragma once

#include <Pathtracing/Shader/OptixRaytracingShader.h>
#include <Pathtracing/Shader/PathtracingShader.cuh>
#include <Pathtracing/Shape/OptixAccelerationStructure.h>

namespace atcg
{
/**
 * @brief A standard pathtracing shader
 */
class PathtracingShader : public OptixRaytracingShader
{
public:
    /**
     * @brief Initialize the shader
     *
     * @param context The optix context
     */
    PathtracingShader(OptixDeviceContext context);

    /**
     * @brief Destructor
     */
    ~PathtracingShader();

    /**
     * @brief Initialize the optix pipeline.
     * Each Optix component has to initialize its part of the raytracing pipeline by defining appropriate entry points
     * and sbt entries.
     *
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

    /**
     * @brief Reset the internal buffers of the shader
     */
    virtual void reset() override;

    /**
     * @brief Set the camera parameters
     *
     * @param camera The camera
     */
    virtual void setCamera(const atcg::ref_ptr<PerspectiveCamera>& camera) override;

    /**
     * @brief Generate rays (trace the image)
     *
     * @param pipeline The pipeline
     * @param sbt The sbt
     * @param output The output tensor
     */
    virtual void generateRays(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                              const atcg::ref_ptr<ShaderBindingTable>& sbt,
                              torch::Tensor& output) override;

private:
    atcg::dref_ptr<PathtracingParams> _launch_params;

    glm::mat4 _inv_camera_view;
    float _fov_y;

    cudaStream_t _stream;

    atcg::ref_ptr<OptixEmitter> _environment_emitter = nullptr;
    atcg::DeviceBuffer<const atcg::EmitterVPtrTable*> _emitter_tables;

    uint32_t _frame_counter        = 0;
    uint32_t _raygen_index         = 0;
    uint32_t _surface_miss_index   = 0;
    uint32_t _occlusion_miss_index = 0;

    atcg::ref_ptr<IASAccelerationStructure> _accel;
};
}    // namespace atcg