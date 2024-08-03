#pragma once

#include <Pathtracing/Shader/CPURaytracingShader.h>
#include <DataStructure/Image.h>
#include <Pathtracing/AccelerationStructure.h>

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

    std::vector<atcg::ref_ptr<Image>> _diffuse_images;
    std::vector<atcg::ref_ptr<Image>> _roughness_images;
    std::vector<atcg::ref_ptr<Image>> _metallic_images;

    std::vector<glm::vec3> _positions;
    std::vector<glm::vec3> _normals;
    std::vector<glm::vec3> _uvs;
    std::vector<glm::u32vec3> _faces;
    std::vector<uint32_t> _mesh_idx;

    atcg::ref_ptr<BVHAccelerationStructure> _accel;
    bool _hasSkybox = false;
    atcg::ref_ptr<Image> _skybox_image;

    std::vector<uint32_t> _horizontalScanLine;
    std::vector<uint32_t> _verticalScanLine;
};
}    // namespace atcg