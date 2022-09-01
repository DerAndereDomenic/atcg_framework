#include <Renderer/CameraController.h>

namespace atcg
{
    CameraController::CameraController(const float& aspect_ratio)
    {
        _camera = std::make_unique<Camera>(aspect_ratio, glm::vec3(0,0,-2));
    }

    void CameraController::onUpdate(float delta_time)
    {

    }

    void CameraController::onEvent(Event& e)
    {

    }
}