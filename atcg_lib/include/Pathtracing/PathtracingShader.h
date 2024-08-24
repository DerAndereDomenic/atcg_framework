#pragma once

#include <Pathtracing/RaytracingShader.h>
#include <DataStructure/Image.h>
#include <Pathtracing/AccelerationStructure.h>
#include <Pathtracing/BSDFModels.h>
#include <Pathtracing/EmitterModels.h>
#include <Pathtracing/PathtracingShader.cuh>

#include <nanort.h>

namespace atcg
{
class PathtracingShader : public RaytracingShader
{
public:
    PathtracingShader();

    ~PathtracingShader();

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

    virtual void reset() override;

    virtual void setCamera(const atcg::ref_ptr<PerspectiveCamera>& camera) override;

    virtual void generateRays(torch::Tensor& output) override;

private:
    glm::mat4 _inv_camera_view;
    float _fov_y;

    uint32_t _frame_counter = 0;
    uint32_t _raygen_idx    = 0;

    std::vector<const EmitterVPtrTable*> _emitter;
    atcg::ref_ptr<EnvironmentEmitter> _environment_emitter;

    torch::Tensor _positions;
    torch::Tensor _normals;
    torch::Tensor _uvs;
    torch::Tensor _faces;
    torch::Tensor _mesh_idx;

    atcg::ref_ptr<IASAccelerationStructure> _accel;

    torch::Tensor _horizontalScanLine;
    torch::Tensor _verticalScanLine;

    std::vector<const BSDFVPtrTable*> _bsdfs;

    atcg::ref_ptr<ShaderBindingTable> _sbt;
    atcg::ref_ptr<RayTracingPipeline> _pipeline;
};
}    // namespace atcg