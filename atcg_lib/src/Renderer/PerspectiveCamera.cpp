#include <Renderer/PerspectiveCamera.h>
#include <glm/gtx/transform.hpp>

namespace atcg
{
    PerspectiveCamera::PerspectiveCamera(const float& aspect_ratio, const glm::vec3& position, const glm::vec3& look_at)
        :_position(position),
         _up(glm::vec3(0,1,0)),
         _look_at(look_at),
         _aspect_ratio(aspect_ratio)
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
        _projection = glm::perspective(glm::radians(60.0f), _aspect_ratio, 0.01f, 100.0f);
    }
}