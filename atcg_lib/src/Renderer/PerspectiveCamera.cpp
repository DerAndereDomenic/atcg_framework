#include <Renderer/PerspectiveCamera.h>

namespace atcg
{
PerspectiveCamera::PerspectiveCamera(const float& aspect_ratio, const glm::vec3& position, const glm::vec3& look_at)
    : _position(position),
      _up(glm::vec3(0, 1, 0)),
      _look_at(look_at),
      _aspect_ratio(aspect_ratio),
      _near(0.01f),
      _far(1000.0f)
{
    recalculateView();
    recalculateProjection();
}

void PerspectiveCamera::recalculateView()
{
    _view = glm::lookAt(_position, _look_at, _up);
}

void PerspectiveCamera::recalculateProjection()
{
    _projection = glm::perspective(glm::radians(60.0f), _aspect_ratio, _near, _far);
}

void PerspectiveCamera::setView(const glm::mat4& view)
{
    _view = view;

    glm::mat4 inv_view = glm::inverse(view);
    _position          = glm::vec3(inv_view[3]);
    _look_at           = _position - glm::vec3(inv_view[2]);
}

}    // namespace atcg