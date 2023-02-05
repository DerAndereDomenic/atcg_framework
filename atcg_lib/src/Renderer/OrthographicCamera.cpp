#include <Renderer/OrthographicCamera.h>
#include <glm/gtx/transform.hpp>

namespace atcg
{
OrthographicCamera::OrthographicCamera(const float& left, const float& right, const float& bottom, const float& top)
    : _left(left)
    , _right(right)
    , _bottom(bottom)
    , _top(top)
{
    recalculateProjection();
}

void OrthographicCamera::recalculateView() {}

void OrthographicCamera::recalculateProjection()
{
    _projection = glm::ortho(_left, _right, _bottom, _top);
}
}    // namespace atcg