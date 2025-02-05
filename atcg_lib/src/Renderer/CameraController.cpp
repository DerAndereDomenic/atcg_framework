#include <Renderer/CameraController.h>

#include <Renderer/VRSystem.h>
#include <openvr.h>

#include <Core/KeyCodes.h>
#include <Core/Application.h>

namespace atcg
{
CameraController::CameraController(const float& aspect_ratio, const glm::vec3& position, const glm::vec3& look_at)
{
    _camera = atcg::make_ref<PerspectiveCamera>(aspect_ratio, position, look_at);
}

CameraController::CameraController(const atcg::ref_ptr<PerspectiveCamera>& camera) : _camera(camera) {}

FocusedController::FocusedController(const float& aspect_ratio) : CameraController(aspect_ratio) {}

FocusedController::FocusedController(const atcg::ref_ptr<PerspectiveCamera>& camera) : CameraController(camera)
{
    _distance = glm::length(_camera->getPosition() - _camera->getLookAt());
}

bool FocusedController::onUpdate(float delta_time)
{
    bool update         = false;
    glm::vec2 mouse_pos = Input::getMousePosition();
    _currentX           = mouse_pos.x;
    _currentY           = mouse_pos.y;

    if(Input::isMouseButtonPressed(ATCG_MOUSE_BUTTON_MIDDLE))
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
            update = true;
        }
    }
    else if(Input::isMouseButtonPressed(ATCG_MOUSE_BUTTON_RIGHT))
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

            update = true;
        }
    }

    _lastX = _currentX;
    _lastY = _currentY;

    return update;
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

FirstPersonController::FirstPersonController(const atcg::ref_ptr<PerspectiveCamera>& camera) : CameraController(camera)
{
}

bool FirstPersonController::onUpdate(float delta_time)
{
    bool update         = false;
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
            update = true;
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

        update = true;
    }
    else if(_pressed_S && !_pressed_W)    // backward
    {
        if(-_velocity_threshold < _velocity_forward && _velocity_forward < _velocity_threshold)
            _velocity_forward = -_velocity_threshold;
        else if(-max_velocity <= _velocity_forward && _velocity_forward < 0.0f)
            _velocity_forward -= delta_velocity;
        else if(0.0f < _velocity_forward)
            _velocity_forward -= deceleration(delta_velocity, _velocity_forward, max_velocity) + delta_velocity;

        update = true;
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

        update = true;
    }
    else if(_pressed_D && !_pressed_A)    // right
    {
        if(-_velocity_threshold < _velocity_right && _velocity_right < _velocity_threshold)
            _velocity_right = _velocity_threshold;
        else if(0.0 < _velocity_right && _velocity_right <= max_velocity)
            _velocity_right += delta_velocity;
        else if(_velocity_right < 0.0)
            _velocity_right -= deceleration(delta_velocity, _velocity_right, max_velocity) - delta_velocity;

        update = true;
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

        update = true;
    }
    else if(_pressed_Q && !_pressed_E)    // down
    {
        if(-_velocity_threshold < _velocity_up && _velocity_up < _velocity_threshold)
            _velocity_up = -_velocity_threshold;
        else if(-max_velocity <= _velocity_up && _velocity_up < 0.0f)
            _velocity_up -= delta_velocity;
        else if(0.0f < _velocity_up)
            _velocity_up -= deceleration(delta_velocity, _velocity_up, max_velocity) + delta_velocity;

        update = true;
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

    if(!Input::isKeyPressed(ATCG_KEY_W)) _pressed_W = false;
    if(!Input::isKeyPressed(ATCG_KEY_A)) _pressed_A = false;
    if(!Input::isKeyPressed(ATCG_KEY_S)) _pressed_S = false;
    if(!Input::isKeyPressed(ATCG_KEY_D)) _pressed_D = false;
    if(!Input::isKeyPressed(ATCG_KEY_Q)) _pressed_Q = false;
    if(!Input::isKeyPressed(ATCG_KEY_E)) _pressed_E = false;

    return update;
}

void FirstPersonController::onEvent(Event* e)
{
    EventDispatcher dispatcher(e);
    dispatcher.dispatch<WindowResizeEvent>(ATCG_BIND_EVENT_FN(FirstPersonController::onWindowResize));
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

bool FirstPersonController::onKeyPressed(KeyPressedEvent* event)
{
    if(event->getKeyCode() == ATCG_KEY_KP_ADD)    // faster
    {
        _speed *= 1.25;
    }

    if(event->getKeyCode() == ATCG_KEY_KP_SUBTRACT)    // slower
    {
        _speed *= 0.8;
    }

    if(event->getKeyCode() == ATCG_KEY_W) _pressed_W = true;
    if(event->getKeyCode() == ATCG_KEY_A) _pressed_A = true;
    if(event->getKeyCode() == ATCG_KEY_S) _pressed_S = true;
    if(event->getKeyCode() == ATCG_KEY_D) _pressed_D = true;
    if(event->getKeyCode() == ATCG_KEY_Q) _pressed_Q = true;
    if(event->getKeyCode() == ATCG_KEY_E) _pressed_E = true;

    return true;
}

bool FirstPersonController::onKeyReleased(KeyReleasedEvent* event)
{
    if(event->getKeyCode() == ATCG_KEY_W) _pressed_W = false;
    if(event->getKeyCode() == ATCG_KEY_A) _pressed_A = false;
    if(event->getKeyCode() == ATCG_KEY_S) _pressed_S = false;
    if(event->getKeyCode() == ATCG_KEY_D) _pressed_D = false;
    if(event->getKeyCode() == ATCG_KEY_Q) _pressed_Q = false;
    if(event->getKeyCode() == ATCG_KEY_E) _pressed_E = false;

    return true;
}

bool FirstPersonController::onMouseButtonPressed(MouseButtonPressedEvent* event)
{
    if(event->getMouseButton() == ATCG_MOUSE_BUTTON_RIGHT)
    {
        const atcg::Application* app = atcg::Application::get();
        glm::ivec2 offset            = app->getViewportPosition();
        int height                   = app->getViewportSize().y;
        glm::vec2 mouse_pos          = glm::vec2(event->getX() - offset.x, height - (event->getY() - offset.y));

        bool in_viewport =
            mouse_pos.x >= 0 && mouse_pos.y >= 0 && mouse_pos.y < height && mouse_pos.x < app->getViewportSize().x;

        if(in_viewport) _clicked_right = true;
    }

    return true;
}

bool FirstPersonController::onMouseButtonReleased(MouseButtonReleasedEvent* event)
{
    if(event->getMouseButton() == ATCG_MOUSE_BUTTON_RIGHT) _clicked_right = false;

    return true;
}


VRController::VRController(const float& aspect_ratio, const float& speed)
    : CameraController(aspect_ratio, glm::vec3(0), glm::vec3(0, 0, 1)),
      _speed(speed)
{
    _cam_left  = atcg::make_ref<PerspectiveCamera>(aspect_ratio);
    _cam_right = atcg::make_ref<PerspectiveCamera>(aspect_ratio);
    _camera    = _cam_left;

    auto [p_left, p_right] = VR::getProjections();
    _cam_left->setProjection(p_left);
    _cam_right->setProjection(p_right);
}

VRController::VRController(const atcg::ref_ptr<PerspectiveCamera>& camera) : CameraController(camera) {}

bool VRController::onUpdate(float delta_time)
{
    bool update          = false;
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

        update = true;
    }
    else if(_pressed_S && !_pressed_W)    // backward
    {
        if(-_velocity_threshold < _velocity_forward && _velocity_forward < _velocity_threshold)
            _velocity_forward = -_velocity_threshold;
        else if(-max_velocity <= _velocity_forward && _velocity_forward < 0.0f)
            _velocity_forward -= delta_velocity;
        else if(0.0f < _velocity_forward)
            _velocity_forward -= deceleration(delta_velocity, _velocity_forward, max_velocity) + delta_velocity;

        update = true;
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

        update = true;
    }
    else if(_pressed_D && !_pressed_A)    // right
    {
        if(-_velocity_threshold < _velocity_right && _velocity_right < _velocity_threshold)
            _velocity_right = _velocity_threshold;
        else if(0.0 < _velocity_right && _velocity_right <= max_velocity)
            _velocity_right += delta_velocity;
        else if(_velocity_right < 0.0)
            _velocity_right -= deceleration(delta_velocity, _velocity_right, max_velocity) - delta_velocity;

        update = true;
    }
    else
    {
        if(-_velocity_threshold < _velocity_right && _velocity_right < _velocity_threshold)
            _velocity_right = 0.0f;
        else
            _velocity_right -= deceleration(delta_velocity, _velocity_right, max_velocity);
    }

    _velocity_forward = glm::clamp(_velocity_forward, -max_velocity, max_velocity);
    _velocity_right   = glm::clamp(_velocity_right, -max_velocity, max_velocity);

    // update camera position
    auto [v_left, v_right] = VR::getInverseViews();
    _cam_left->setView(glm::inverse(v_left));
    _cam_right->setView(glm::inverse(v_right));

    glm::vec3 forward_left = -v_left[2];

    forward_left[1]          = 0.0f;    // only horizontal movement
    glm::vec3 upDirection    = glm::vec3(0, 1, 0);
    glm::vec3 rightDirection = glm::normalize(glm::cross(forward_left, upDirection));

    glm::vec3 total_velocity = _speed * (forward_left * _velocity_forward + rightDirection * _velocity_right);
    glm::vec3 current_offset = VR::getOffset();
    current_offset += total_velocity * delta_time;


    // The trigger is currently down
    if(_trigger_pressed)
    {
        glm::mat4 pose        = VR::getDevicePose(_device_index);
        _controller_position  = glm::vec3(pose[3]);
        _controller_direction = -pose[2];

        float t = -_controller_position.y / _controller_direction.y;

        if(std::abs(_controller_direction.y) > 1e-5f && t > 0.0f)
        {
            _controller_intersection = _controller_position + t * _controller_direction;
        }
        else
        {
            _controller_intersection = _controller_position + 1000.0f * _controller_direction;
        }

        update = true;
    }

    // Update camera position
    if(_trigger_release)
    {
        _trigger_release = false;

        float t = -_controller_position.y / _controller_direction.y;

        if(std::abs(_controller_direction.y) > 1e-5f && t > 0.0f)
        {
            auto rl_pos    = atcg::VR::getPosition();
            rl_pos.y       = 0;
            current_offset = _controller_position + t * _controller_direction - rl_pos;
        }

        update = true;
    }

    VR::setOffset(current_offset);

    if(!Input::isKeyPressed(ATCG_KEY_W)) _pressed_W = false;
    if(!Input::isKeyPressed(ATCG_KEY_A)) _pressed_A = false;
    if(!Input::isKeyPressed(ATCG_KEY_S)) _pressed_S = false;
    if(!Input::isKeyPressed(ATCG_KEY_D)) _pressed_D = false;

    return update;
}

void VRController::onEvent(Event* e)
{
    EventDispatcher dispatcher(e);
    dispatcher.dispatch<WindowResizeEvent>(ATCG_BIND_EVENT_FN(VRController::onWindowResize));
    dispatcher.dispatch<KeyPressedEvent>(ATCG_BIND_EVENT_FN(VRController::onKeyPressed));
    dispatcher.dispatch<KeyReleasedEvent>(ATCG_BIND_EVENT_FN(VRController::onKeyReleased));
    dispatcher.dispatch<VRButtonPressedEvent>(ATCG_BIND_EVENT_FN(VRController::onVRButtonPressed));
    dispatcher.dispatch<VRButtonReleasedEvent>(ATCG_BIND_EVENT_FN(VRController::onVRButtonReleased));
}

bool VRController::onWindowResize(WindowResizeEvent* event)
{
    return false;
}

bool VRController::onKeyPressed(KeyPressedEvent* event)
{
    if(event->getKeyCode() == ATCG_KEY_KP_ADD)    // faster
    {
        _speed *= 1.25;
    }

    if(event->getKeyCode() == ATCG_KEY_KP_SUBTRACT)    // slower
    {
        _speed *= 0.8;
    }

    if(event->getKeyCode() == ATCG_KEY_W) _pressed_W = true;
    if(event->getKeyCode() == ATCG_KEY_A) _pressed_A = true;
    if(event->getKeyCode() == ATCG_KEY_S) _pressed_S = true;
    if(event->getKeyCode() == ATCG_KEY_D) _pressed_D = true;

    return true;
}

bool VRController::onKeyReleased(KeyReleasedEvent* event)
{
    if(event->getKeyCode() == ATCG_KEY_W) _pressed_W = false;
    if(event->getKeyCode() == ATCG_KEY_A) _pressed_A = false;
    if(event->getKeyCode() == ATCG_KEY_S) _pressed_S = false;
    if(event->getKeyCode() == ATCG_KEY_D) _pressed_D = false;

    return true;
}

bool VRController::onVRButtonPressed(VRButtonPressedEvent* event)
{
    auto role = VR::getDeviceRole(event->getDeviceIndex());
    if(event->getVRButton() == vr::EVRButtonId::k_EButton_SteamVR_Trigger && role == VRSystem::Role::RIGHT_HAND)
    {
        _trigger_pressed = true;
        _device_index    = event->getDeviceIndex();
    }
    return true;
}

bool VRController::onVRButtonReleased(VRButtonReleasedEvent* event)
{
    auto role = VR::getDeviceRole(event->getDeviceIndex());
    if(event->getVRButton() == vr::EVRButtonId::k_EButton_SteamVR_Trigger && role == VRSystem::Role::RIGHT_HAND)
    {
        _trigger_release = true;
        _trigger_pressed = false;
    }
    return true;
}
}    // namespace atcg