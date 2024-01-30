#include <Renderer/CameraController.h>

#include <GLFW/glfw3.h>

namespace atcg
{
CameraController::CameraController(const float& aspect_ratio, const glm::vec3& position, const glm::vec3& look_at)
{
    _camera = atcg::make_ref<PerspectiveCamera>(aspect_ratio, position, look_at);
}

CameraController::CameraController(const atcg::ref_ptr<PerspectiveCamera>& camera) : _camera(camera) {}

FocusedController::FocusedController(const float& aspect_ratio) : CameraController(aspect_ratio) {}

FocusedController::FocusedController(const atcg::ref_ptr<Camera>& camera) : CameraController(camera)
{
    _distance = glm::length(_camera->getPosition() - _camera->getLookAt());
}

void FocusedController::onUpdate(float delta_time)
{
    glm::vec2 mouse_pos = Input::getMousePosition();
    _currentX           = mouse_pos.x;
    _currentY           = mouse_pos.y;

    if(Input::isMouseButtonPressed(GLFW_MOUSE_BUTTON_MIDDLE))
    {
        float offsetX = _lastX - _currentX;
        float offsetY = _lastY - _currentY;

        if(offsetX != 0 || offsetY != 0)
        {
            float pitchDelta = offsetY * /*delta_time */ _rotation_speed * _camera->getAspectRatio();
            float yawDelta   = -offsetX * /*delta_time */ _rotation_speed;

            glm::vec3 forward = glm::normalize(_camera->getPosition() - _camera->getLookAt());

            glm::vec3 rightDirection = glm::cross(forward, _camera->getUp());

            glm::quat q = glm::normalize(
                glm::cross(glm::angleAxis(-pitchDelta, rightDirection), glm::angleAxis(-yawDelta, _camera->getUp())));
            forward = glm::rotate(q, forward);

            _camera->setPosition(_camera->getLookAt() + _distance * forward);
        }
    }
    else if(Input::isMouseButtonPressed(GLFW_MOUSE_BUTTON_RIGHT))
    {
        float offsetX = _lastX - _currentX;
        float offsetY = _lastY - _currentY;

        if(offsetX != 0 || offsetY != 0)
        {
            float yDelta = 0.1f * -offsetY * /*delta_time */ _rotation_speed * _distance * _camera->getAspectRatio();
            float xDelta = 0.1f * -offsetX * /*delta_time */ _rotation_speed * _distance;

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

void FocusedController::onEvent(Event* e)
{
    EventDispatcher dispatcher(e);
    dispatcher.dispatch<MouseScrolledEvent>(ATCG_BIND_EVENT_FN(FocusedController::onMouseZoom));
    dispatcher.dispatch<WindowResizeEvent>(ATCG_BIND_EVENT_FN(FocusedController::onWindowResize));
}

bool FocusedController::onMouseZoom(MouseScrolledEvent* event)
{
    float offset = event->getYOffset();

    _distance *= glm::exp2(-offset * _zoom_speed);
    glm::vec3 back_dir = glm::normalize(_camera->getPosition() - _camera->getLookAt());
    _camera->setPosition(_camera->getLookAt() + back_dir * _distance);

    return false;
}

bool FocusedController::onWindowResize(WindowResizeEvent* event)
{
    float aspect_ratio = (float)event->getWidth() / (float)event->getHeight();
    _camera->setAspectRatio(aspect_ratio);
    return false;
}


FirstPersonController::FirstPersonController(const float& aspect_ratio,
                                             const glm::vec3& position,
                                             const glm::vec3& view_direction,
                                             const float& speed)
    : CameraController(aspect_ratio, position, position + view_direction),
      _speed(speed)
{
}

FirstPersonController::FirstPersonController(const atcg::ref_ptr<Camera>& camera) : CameraController(camera) {}

void FirstPersonController::onUpdate(float delta_time)
{
    glm::vec2 mouse_pos = Input::getMousePosition();
    _currentX           = mouse_pos.x;
    _currentY           = mouse_pos.y;

    if(_clicked_right)
    {
        float offsetX = _lastX - _currentX;
        float offsetY = _lastY - _currentY;

        if(offsetX != 0 || offsetY != 0)
        {
            float pitchDelta = offsetY * _rotation_speed * _camera->getAspectRatio();
            float yawDelta   = offsetX * _rotation_speed;

            glm::vec3 forward        = glm::normalize(_camera->getLookAt() - _camera->getPosition());
            glm::vec3 rightDirection = glm::cross(forward, _camera->getUp());

            glm::quat q = glm::normalize(
                glm::cross(glm::angleAxis(pitchDelta, rightDirection), glm::angleAxis(yawDelta, _camera->getUp())));
            forward = glm::rotate(q, forward);

            _camera->setLookAt(_camera->getPosition() + forward);
        }
    }

    float delta_velocity = _acceleration * delta_time;
    float max_velocity   = _max_velocity;
    auto deceleration    = [](float delta_velocity, float current_velocity, float max_velocity)
    {
        float relative_velocity = current_velocity / max_velocity;
        float deceleration_factor =
            delta_velocity * (6.0f * glm::sign(relative_velocity) * (relative_velocity * relative_velocity + 0.1));
        if(deceleration_factor / current_velocity > 1.0f)
            return current_velocity;
        else
            return deceleration_factor;
    };

    if(_pressed_W && !_pressed_S)    // forward
    {
        if(-_velocity_threshold < _velocity_forward && _velocity_forward < _velocity_threshold)
            _velocity_forward = _velocity_threshold;
        else if(0.0f < _velocity_forward && _velocity_forward <= max_velocity)
            _velocity_forward += delta_velocity;
        else if(_velocity_forward < 0.0f)
            _velocity_forward -= deceleration(delta_velocity, _velocity_forward, max_velocity) - delta_velocity;
    }
    else if(_pressed_S && !_pressed_W)    // backward
    {
        if(-_velocity_threshold < _velocity_forward && _velocity_forward < _velocity_threshold)
            _velocity_forward = -_velocity_threshold;
        else if(-max_velocity <= _velocity_forward && _velocity_forward < 0.0f)
            _velocity_forward -= delta_velocity;
        else if(0.0f < _velocity_forward)
            _velocity_forward -= deceleration(delta_velocity, _velocity_forward, max_velocity) + delta_velocity;
    }
    else
    {
        if(-_velocity_threshold < _velocity_forward && _velocity_forward < _velocity_threshold)
            _velocity_forward = 0.0f;
        else
            _velocity_forward -= deceleration(delta_velocity, _velocity_forward, max_velocity);
    }

    if(_pressed_A && !_pressed_D)    // left
    {
        if(-_velocity_threshold < _velocity_right && _velocity_right < _velocity_threshold)
            _velocity_right = -_velocity_threshold;
        else if(-max_velocity <= _velocity_right && _velocity_right < 0.0f)
            _velocity_right -= delta_velocity;
        else if(0.0f < _velocity_right)
            _velocity_right -= deceleration(delta_velocity, _velocity_right, max_velocity) + delta_velocity;
    }
    else if(_pressed_D && !_pressed_A)    // right
    {
        if(-_velocity_threshold < _velocity_right && _velocity_right < _velocity_threshold)
            _velocity_right = _velocity_threshold;
        else if(0.0 < _velocity_right && _velocity_right <= max_velocity)
            _velocity_right += delta_velocity;
        else if(_velocity_right < 0.0)
            _velocity_right -= deceleration(delta_velocity, _velocity_right, max_velocity) - delta_velocity;
    }
    else
    {
        if(-_velocity_threshold < _velocity_right && _velocity_right < _velocity_threshold)
            _velocity_right = 0.0f;
        else
            _velocity_right -= deceleration(delta_velocity, _velocity_right, max_velocity);
    }

    if(_pressed_E && !_pressed_Q)    // up
    {
        if(-_velocity_threshold < _velocity_up && _velocity_up < _velocity_threshold)
            _velocity_up = _velocity_threshold;
        else if(0.0f < _velocity_up && _velocity_up <= max_velocity)
            _velocity_up += delta_velocity;
        else if(_velocity_up < 0.0)
            _velocity_up -= deceleration(delta_velocity, _velocity_up, max_velocity) - delta_velocity;
    }
    else if(_pressed_Q && !_pressed_E)    // down
    {
        if(-_velocity_threshold < _velocity_up && _velocity_up < _velocity_threshold)
            _velocity_up = -_velocity_threshold;
        else if(-max_velocity <= _velocity_up && _velocity_up < 0.0f)
            _velocity_up -= delta_velocity;
        else if(0.0f < _velocity_up)
            _velocity_up -= deceleration(delta_velocity, _velocity_up, max_velocity) + delta_velocity;
    }
    else
    {
        if(-_velocity_threshold < _velocity_up && _velocity_up < _velocity_threshold)
            _velocity_up = 0.0f;
        else
            _velocity_up -= deceleration(delta_velocity, _velocity_up, max_velocity);
    }

    _velocity_forward = glm::clamp(_velocity_forward, -max_velocity, max_velocity);
    _velocity_right   = glm::clamp(_velocity_right, -max_velocity, max_velocity);
    _velocity_up      = glm::clamp(_velocity_up, -max_velocity, max_velocity);

    // update camera position
    glm::vec3 forwardDirection = _camera->getLookAt() - _camera->getPosition();
    forwardDirection[1]        = 0.0f;    // only horizontal movement
    forwardDirection           = glm::normalize(forwardDirection);
    glm::vec3 upDirection      = _camera->getUp();
    glm::vec3 rightDirection   = glm::normalize(glm::cross(forwardDirection, upDirection));

    glm::vec3 total_velocity =
        _speed * (forwardDirection * _velocity_forward + rightDirection * _velocity_right + upDirection * _velocity_up);
    _camera->setPosition(_camera->getPosition() + total_velocity * delta_time);
    _camera->setLookAt(_camera->getLookAt() + total_velocity * delta_time);

    // update mouse position
    _lastX = _currentX;
    _lastY = _currentY;

    if(!Input::isKeyPressed(GLFW_KEY_W)) _pressed_W = false;
    if(!Input::isKeyPressed(GLFW_KEY_A)) _pressed_A = false;
    if(!Input::isKeyPressed(GLFW_KEY_S)) _pressed_S = false;
    if(!Input::isKeyPressed(GLFW_KEY_D)) _pressed_D = false;
    if(!Input::isKeyPressed(GLFW_KEY_Q)) _pressed_Q = false;
    if(!Input::isKeyPressed(GLFW_KEY_E)) _pressed_E = false;
}

void FirstPersonController::onEvent(Event* e)
{
    EventDispatcher dispatcher(e);
    dispatcher.dispatch<WindowResizeEvent>(ATCG_BIND_EVENT_FN(FirstPersonController::onWindowResize));
    dispatcher.dispatch<MouseMovedEvent>(ATCG_BIND_EVENT_FN(FirstPersonController::onMouseMove));
    dispatcher.dispatch<KeyPressedEvent>(ATCG_BIND_EVENT_FN(FirstPersonController::onKeyPressed));
    dispatcher.dispatch<KeyReleasedEvent>(ATCG_BIND_EVENT_FN(FirstPersonController::onKeyReleased));
    dispatcher.dispatch<MouseButtonPressedEvent>(ATCG_BIND_EVENT_FN(FirstPersonController::onMouseButtonPressed));
    dispatcher.dispatch<MouseButtonReleasedEvent>(ATCG_BIND_EVENT_FN(FirstPersonController::onMouseButtonReleased));
}

bool FirstPersonController::onWindowResize(WindowResizeEvent* event)
{
    float aspect_ratio = (float)event->getWidth() / (float)event->getHeight();
    _camera->setAspectRatio(aspect_ratio);
    return false;
}

bool FirstPersonController::onMouseMove(MouseMovedEvent* event)
{
    if(Input::isMouseButtonPressed(GLFW_MOUSE_BUTTON_RIGHT))
    {
        _clicked_right = true;
    }

    return false;
}

bool FirstPersonController::onKeyPressed(KeyPressedEvent* event)
{
    if(event->getKeyCode() == GLFW_KEY_KP_ADD)    // faster
    {
        _speed *= 1.25;
    }

    if(event->getKeyCode() == GLFW_KEY_KP_SUBTRACT)    // slower
    {
        _speed *= 0.8;
    }

    if(event->getKeyCode() == GLFW_KEY_W) _pressed_W = true;
    if(event->getKeyCode() == GLFW_KEY_A) _pressed_A = true;
    if(event->getKeyCode() == GLFW_KEY_S) _pressed_S = true;
    if(event->getKeyCode() == GLFW_KEY_D) _pressed_D = true;
    if(event->getKeyCode() == GLFW_KEY_Q) _pressed_Q = true;
    if(event->getKeyCode() == GLFW_KEY_E) _pressed_E = true;

    return true;
}

bool FirstPersonController::onKeyReleased(KeyReleasedEvent* event)
{
    if(event->getKeyCode() == GLFW_KEY_W) _pressed_W = false;
    if(event->getKeyCode() == GLFW_KEY_A) _pressed_A = false;
    if(event->getKeyCode() == GLFW_KEY_S) _pressed_S = false;
    if(event->getKeyCode() == GLFW_KEY_D) _pressed_D = false;
    if(event->getKeyCode() == GLFW_KEY_Q) _pressed_Q = false;
    if(event->getKeyCode() == GLFW_KEY_E) _pressed_E = false;

    return true;
}

bool FirstPersonController::onMouseButtonPressed(MouseButtonPressedEvent* event)
{
    if(event->getMouseButton() == GLFW_MOUSE_BUTTON_RIGHT) _clicked_right = true;

    return true;
}

bool FirstPersonController::onMouseButtonReleased(MouseButtonReleasedEvent* event)
{
    if(event->getMouseButton() == GLFW_MOUSE_BUTTON_RIGHT) _clicked_right = false;

    return true;
}

}    // namespace atcg