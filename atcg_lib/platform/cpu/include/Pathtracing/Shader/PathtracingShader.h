#pragma once

#include <Pathtracing/Shader/CPURaytracingShader.h>
#include <DataStructure/Image.h>
#include <Pathtracing/AccelerationStructure.h>
#include <Pathtracing/BSDF/BSDFModels.h>

#include <nanort.h>

namespace atcg
{
class PathtracingShader : public CPURaytracingShader
{
public:
    PathtracingShader();

    ~PathtracingShader();

    virtual void initializePipeline() override;

    virtual void reset() override;

    virtual void setCamera(const atcg::ref_ptr<PerspectiveCamera>& camera) override;

    virtual void generateRays(torch::Tensor& output) override;

private:
    glm::vec3 read_image(const atcg::ref_ptr<Image>& image, const glm::vec2& uv);

    glm::mat4 _inv_camera_view;
    float _fov_y;

    uint32_t _frame_counter = 0;

    std::vector<atcg::ref_ptr<CPUBSDF>> _bsdfs;

    torch::Tensor _positions;
    torch::Tensor _normals;
    torch::Tensor _uvs;
    torch::Tensor _faces;
    torch::Tensor _mesh_idx;

    atcg::ref_ptr<BVHAccelerationStructure> _accel;
    bool _hasSkybox = false;
    atcg::ref_ptr<Image> _skybox_image;

    torch::Tensor _horizontalScanLine;
    torch::Tensor _verticalScanLine;
};
}    // namespace atcg