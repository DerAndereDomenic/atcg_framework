#include <Renderer/PerspectiveCamera.h>

namespace atcg
{
PerspectiveCamera::PerspectiveCamera(const CameraExtrinsics& extrinsics, const CameraIntrinsics& intrinsics)
{
    _extrinsics = extrinsics;
    _intrinsics = intrinsics;
}

atcg::ref_ptr<Camera> PerspectiveCamera::copy() const
{
    atcg::ref_ptr<PerspectiveCamera> camera = atcg::make_ref<PerspectiveCamera>(_extrinsics, _intrinsics);

    return camera;
}

glm::vec3 PerspectiveCamera::transformToCameraSpace(const glm::vec3& point) const
{
    glm::vec4 p = _extrinsics.extrinsicMatrix() * glm::vec4(point, 1.0f);
    return glm::vec3(p);
}

glm::vec3 PerspectiveCamera::transformToNormalizedDeviceCoordinates(const glm::vec3& point) const
{
    glm::vec4 p = _intrinsics.projection() * glm::vec4(transformToCameraSpace(point), 1.0f);
    return glm::vec3(p) / p.w;
}

bool PerspectiveCamera::isPointInFrustum(const glm::vec3& point) const
{
    glm::vec3 ndc = transformToNormalizedDeviceCoordinates(point);
    return ndc.x >= -1.0f && ndc.x <= 1.0f && ndc.y >= -1.0f && ndc.y <= 1.0f && ndc.z >= -1.0f && ndc.z <= 1.0f;
}

}    // namespace atcg