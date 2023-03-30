#include <Renderer/CameraController.h>

#include <Core/Input.h>
#include <Core/API.h>

#include <iostream>
#include <GLFW/glfw3.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

namespace atcg
{
CameraController::CameraController(const float& aspect_ratio)
{
    _camera   = atcg::make_ref<PerspectiveCamera>(aspect_ratio, glm::vec3(0, 0, -1));
    _distance = 1;
}

void CameraController::onUpdate(float delta_time)
{
    if(Input::isMouseButtonPressed(GLFW_MOUSE_BUTTON_1))
    {
        float offsetX = _lastX - _currentX;
        float offsetY = _lastY - _currentY;

        if(offsetX != 0 || offsetY != 0)
        {
            float pitchDelta = offsetY * /*delta_time */ _parameters.rotation_speed * _camera->getAspectRatio();
            float yawDelta   = -offsetX * /*delta_time */ _parameters.rotation_speed;

            glm::vec3 forward = glm::normalize(_camera->getPosition() - _camera->getLookAt());

            glm::vec3 rightDirection = glm::cross(forward, _camera->getUp());

            glm::quat q = glm::normalize(
                glm::cross(glm::angleAxis(-pitchDelta, rightDirection), glm::angleAxis(-yawDelta, _camera->getUp())));
            forward = glm::rotate(q, forward);

            _camera->setPosition(_camera->getLookAt() + _distance * forward);
        }
    }
    else if(Input::isMouseButtonPressed(GLFW_MOUSE_BUTTON_2))
    {
        float offsetX = _lastX - _currentX;
        float offsetY = _lastY - _currentY;

        if(offsetX != 0 || offsetY != 0)
        {
            float yDelta = -offsetY * /*delta_time */ _parameters.rotation_speed * _camera->getAspectRatio();
            float xDelta = -offsetX * /*delta_time */ _parameters.rotation_speed;

            glm::vec3 forward = glm::normalize(_camera->getPosition() - _camera->getLookAt());

            glm::vec3 up_local       = glm::vec3(glm::inverse(_camera->getView())[1]);
            glm::vec3 rightDirection = glm::cross(forward, up_local);

            glm::vec3 tangent = xDelta * rightDirection + yDelta * up_local;

            _camera->setPosition(_camera->getPosition() + tangent);
            _camera->setLookAt(_camera->getLookAt() + tangent);
        }
    }

    _lastX = _currentX;
    _lastY = _currentY;
}

void CameraController::onEvent(Event* e)
{
    EventDispatcher dispatcher(e);
    dispatcher.dispatch<MouseScrolledEvent>(ATCG_BIND_EVENT_FN(CameraController::onMouseZoom));
    dispatcher.dispatch<WindowResizeEvent>(ATCG_BIND_EVENT_FN(CameraController::onWindowResize));
    dispatcher.dispatch<MouseMovedEvent>(ATCG_BIND_EVENT_FN(CameraController::onMouseMove));
}

bool CameraController::onMouseZoom(MouseScrolledEvent* event)
{
    float offset = event->getYOffset();

    _distance *= glm::exp2(-offset * _parameters.zoom_speed);
    glm::vec3 back_dir = glm::normalize(_camera->getPosition() - _camera->getLookAt());
    _camera->setPosition(_camera->getLookAt() + back_dir * _distance);

    return false;
}

bool CameraController::onWindowResize(WindowResizeEvent* event)
{
    float aspect_ratio = (float)event->getWidth() / (float)event->getHeight();
    _camera->setAspectRatio(aspect_ratio);
    return false;
}

bool CameraController::onMouseMove(MouseMovedEvent* event)
{
    _currentX = event->getX();
    _currentY = event->getY();

    return false;
}
}    // namespace atcg