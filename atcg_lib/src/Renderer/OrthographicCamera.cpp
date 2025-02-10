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

atcg::ref_ptr<Camera> OrthographicCamera::copy() const
{
    return atcg::make_ref<OrthographicCamera>(_left, _right, _bottom, _top);
}
}    // namespace atcg