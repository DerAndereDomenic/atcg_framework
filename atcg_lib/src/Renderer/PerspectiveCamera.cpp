#include <Renderer/PerspectiveCamera.h>

namespace atcg
{
PerspectiveCamera::PerspectiveCamera(const float& aspect_ratio, const glm::vec3& position, const glm::vec3& look_at)
    : _position(position),
      _up(glm::vec3(0, 1, 0)),
      _look_at(look_at),
      _aspect_ratio(aspect_ratio),
      _fovy(60.0f),
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
    _projection = glm::perspective(glm::radians(_fovy), _aspect_ratio, _near, _far);
}

void PerspectiveCamera::setView(const glm::mat4& view)
{
    _view = view;

    glm::mat4 inv_view = glm::inverse(view);
    _position          = glm::vec3(inv_view[3]);
    _look_at           = _position - glm::vec3(inv_view[2]);
}

void PerspectiveCamera::setFromTransform(const glm::mat4& transform)
{
    float scale_x = glm::length(glm::vec3(transform[0]));
    float scale_y = glm::length(glm::vec3(transform[1]));
    float scale_z = glm::length(glm::vec3(transform[2]));

    _position = transform[3];
    _look_at  = _position - glm::vec3(transform[2] / scale_y);

    _view = glm::inverse(transform * glm::scale(glm::vec3(1.0f / scale_x, 1.0f / scale_y, 1.0f / scale_z)));

    _aspect_ratio = scale_x / scale_y;

    recalculateProjection();
}

}    // namespace atcg