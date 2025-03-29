#include <Renderer/PerspectiveCamera.h>

namespace atcg
{
PerspectiveCamera::PerspectiveCamera(const CameraExtrinsics& extrinsics, const CameraIntrinsics& intrinsics)
{
    _extrinsics = extrinsics;
    _intrinsics = intrinsics;
}

void PerspectiveCamera::setFromTransform(const glm::mat4& transform)
{
    float scale_x = glm::length(glm::vec3(transform[0]));
    float scale_y = glm::length(glm::vec3(transform[1]));
    float scale_z = glm::length(glm::vec3(transform[2]));

    _extrinsics.setPosition(transform[3]);
    _extrinsics.setTarget(_extrinsics.position() - glm::vec3(transform[2] / scale_y));

    _extrinsics.setExtrinsicMatrix(
        glm::inverse(transform * glm::scale(glm::vec3(1.0f / scale_x, 1.0f / scale_y, 1.0f / scale_z))));

    _intrinsics.setAspectRatio(scale_x / scale_y);

    _intrinsics.setFOV(glm::degrees(2.0f * glm::atan(scale_z)));
}

glm::mat4 PerspectiveCamera::getAsTransform() const
{
    return glm::inverse(_extrinsics.extrinsicMatrix()) *
           glm::scale(glm::vec3(_intrinsics.aspectRatio(), 1.0f, glm::tan(glm::radians(_intrinsics.FOV()) / 2.0f)));
}

atcg::ref_ptr<Camera> PerspectiveCamera::copy() const
{
    atcg::ref_ptr<PerspectiveCamera> camera = atcg::make_ref<PerspectiveCamera>(_extrinsics, _intrinsics);

    return camera;
}

}    // namespace atcg