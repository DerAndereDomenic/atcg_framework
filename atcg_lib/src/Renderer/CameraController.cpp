#include <Renderer/CameraController.h>
#include <Core/API.h>
#include <iostream>

namespace atcg
{
    CameraController::CameraController(const float& aspect_ratio)
    {
        _camera = std::make_unique<Camera>(aspect_ratio, glm::vec3(0,0,-1));
        _distance = 1;
    }

    void CameraController::onUpdate(float delta_time)
    {

    }

    void CameraController::onEvent(Event& e)
    {
        EventDispatcher dispatcher(e);
        dispatcher.dispatch<MouseScrolledEvent>(ATCG_BIND_EVENT_FN(CameraController::onMouseZoom));
        dispatcher.dispatch<WindowResizeEvent>(ATCG_BIND_EVENT_FN(CameraController::onWindowResize));
    }

    bool CameraController::onMouseZoom(MouseScrolledEvent& event)
    {
        float offset = event.getYOffset();
        
        _distance *= glm::exp2(-offset * _parameters.zoom_speed);
        glm::vec3 back_dir = glm::normalize(_camera->getPosition());
        _camera->setPosition(back_dir * _distance);

        return false;
    }

    bool CameraController::onWindowResize(WindowResizeEvent& event)
    {
        float aspect_ratio = (float)event.getWidth() / (float)event.getHeight();
        _camera->setAspectRatio(aspect_ratio);
        return false;
    }
}