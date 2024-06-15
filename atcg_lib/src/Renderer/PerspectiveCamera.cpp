#include <Renderer/PerspectiveCamera.h>

namespace atcg
{
PerspectiveCamera::PerspectiveCamera(const float& aspect_ratio, const glm::vec3& position, const glm::vec3& look_at)
    : _position(position),
      _up(glm::vec3(0, 1, 0)),
      _look_at(look_at),
      _optical_center(glm::vec2(0)),
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
    // _up   = glm::column(_view, 1);
}

void PerspectiveCamera::recalculateProjection()
{
    _projection       = glm::perspective(glm::radians(_fovy), _aspect_ratio, _near, _far);
    _projection[2][0] = _optical_center.x;
    _projection[2][1] = _optical_center.y;
}

void PerspectiveCamera::setProjection(const glm::mat4& projection)
{
    _projection     = projection;
    _aspect_ratio   = _projection[1][1] / _projection[0][0];
    _fovy           = glm::degrees(2.0f * glm::atan(1.0f / _projection[1][1]));
    _optical_center = glm::vec2(_projection[2][0], _projection[2][1]);
    _near           = -_projection[3][2] / (_projection[2][2] - 1.0f);
    _far            = -_projection[3][2] / (_projection[2][2] + 1.0f);
}

void PerspectiveCamera::setView(const glm::mat4& view)
{
    _view = view;

    glm::mat4 inv_view = glm::inverse(view);
    _position          = glm::vec3(inv_view[3]);
    _look_at           = _position - glm::vec3(inv_view[2]);
    // _up                = glm::column(_view, 1);
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

    _fovy = glm::degrees(2.0f * glm::atan(scale_z));

    recalculateProjection();
}

glm::mat4 PerspectiveCamera::getAsTransform() const
{
    return glm::inverse(_view) * glm::scale(glm::vec3(_aspect_ratio, 1.0f, glm::tan(glm::radians(_fovy) / 2.0f)));
}

}    // namespace atcg