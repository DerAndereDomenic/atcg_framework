#include <Renderer/OrthographicCamera.h>

namespace atcg
{
OrthographicCamera::OrthographicCamera(const float& left, const float& right, const float& bottom, const float& top)
    : _left(left),
      _right(right),
      _bottom(bottom),
      _top(top)
{
    recalculateProjection();
}

void OrthographicCamera::recalculateView() {}

void OrthographicCamera::recalculateProjection()
{
    _projection = glm::ortho(_left, _right, _bottom, _top);
}


glm::vec3 OrthographicCamera::transformToCameraSpace(const glm::vec3& point) const
{
    return point;
}

glm::vec3 OrthographicCamera::transformToNormalizedDeviceCoordinates(const glm::vec3& point) const
{
    glm::vec4 p = _projection * glm::vec4(transformToCameraSpace(point), 1.0f);
    return glm::vec3(p) / p.w;
}

bool OrthographicCamera::isPointInFrustum(const glm::vec3& point) const
{
    glm::vec3 ndc_point = transformToNormalizedDeviceCoordinates(point);
    return glm::abs(ndc_point.x) <= 1.0f && glm::abs(ndc_point.y) <= 1.0f && glm::abs(ndc_point.z) <= 1.0f;
}

atcg::ref_ptr<Camera> OrthographicCamera::copy() const
{
    return atcg::make_ref<OrthographicCamera>(_left, _right, _bottom, _top);
}
}    // namespace atcg