#pragma once

#include <Renderer/PerspectiveCamera.h>
#include <Events/Event.h>
#include <Events/MouseEvent.h>
#include <Events/KeyEvent.h>
#include <Events/WindowEvent.h>
#include <Events/VREvent.h>

#include <Core/Memory.h>

namespace atcg
{
/**
 * @brief A class to model camera movement (only for perspective camera for now)
 * Scroll to zoom
 * Drag to rotate
 */
class CameraController
{
public:
    /**
     * @brief Create a camera controller with a given camera.
     *
     * @param camera The camera
     */
    CameraController(const atcg::ref_ptr<PerspectiveCamera>& camera);

    /**
     * @brief Gets called every frame
     *
     * @param delta_time Time since last frame
     *
     * @return If the camera was moved
     */
    virtual bool onUpdate(float delta_time) = 0;

    /**
     * @brief Handles events
     *
     * @param e The event
     */
    virtual void onEvent(Event* e) = 0;

    /**
     * @brief Get the Camera object
     *
     * @return const atcg::scope_ptr<Camera>& The camera
     */
    ATCG_INLINE const atcg::ref_ptr<PerspectiveCamera>& getCamera() const { return _camera; }

    /**
     * @brief Set the Camera object
     *
     * @param camera The camera
     */
    ATCG_INLINE void setCamera(const atcg::ref_ptr<PerspectiveCamera>& camera) { _camera = camera; }

protected:
    atcg::ref_ptr<PerspectiveCamera> _camera;
};

class FocusedController : public CameraController
{
public:
    /**
     * @brief Construct a new First Person Camera object
     *
     * @param camera The camera
     */
    FocusedController(const atcg::ref_ptr<PerspectiveCamera>& camera);

    /**
     * @brief Gets called every frame
     *
     * @param delta_time Time since last frame
     *
     * @return If the camera was moved
     */
    virtual bool onUpdate(float delta_time) override;

    /**
     * @brief Handles events
     *
     * @param e The event
     */
    virtual void onEvent(Event* e) override;

private:
    bool onMouseZoom(MouseScrolledEvent* event);
    bool onWindowResize(WindowResizeEvent* event);

    float _distance       = 1.0f;
    float _zoom_speed     = 0.25f;
    float _rotation_speed = 0.005f;
    float _lastX = 0, _lastY = 0;
    float _currentX = 0, _currentY = 0;
};

class FirstPersonController : public CameraController
{
public:
    /**
     * @brief Construct a new First Person Camera object
     *
     * @param camera The camera
     */
    FirstPersonController(const atcg::ref_ptr<PerspectiveCamera>& camera);

    /**
     * @brief Gets called every frame
     *
     * @param delta_time Time since last frame
     *
     * @return If the camera was moved
     */
    virtual bool onUpdate(float delta_time) override;

    /**
     * @brief Handles events
     *
     * @param e The event
     */
    virtual void onEvent(Event* e) override;

private:
    bool onWindowResize(WindowResizeEvent* event);
    bool onKeyPressed(KeyPressedEvent* event);
    bool onKeyReleased(KeyReleasedEvent* event);
    bool onMouseButtonPressed(MouseButtonPressedEvent* event);
    bool onMouseButtonReleased(MouseButtonReleasedEvent* event);

    float _speed              = 1.0f;
    float _velocity_forward   = 0.0f;
    float _velocity_right     = 0.0f;
    float _velocity_up        = 0.0f;
    float _velocity_threshold = 0.3f;
    float _max_velocity       = 3.0f;
    float _acceleration       = 5.0f;
    float _rotation_speed     = 0.005f;
    float _lastX = 0, _lastY = 0;
    float _currentX = 0, _currentY = 0;

    bool _pressed_W     = false;
    bool _pressed_A     = false;
    bool _pressed_S     = false;
    bool _pressed_D     = false;
    bool _pressed_Q     = false;
    bool _pressed_E     = false;
    bool _clicked_right = false;
};

class VRController : public CameraController
{
public:
    /**
     * @brief Construct a new First Person Camera object
     *
     * @param camera_left The camera of the left eye
     * @param camera_right The camera of the right eye
     */
    VRController(const atcg::ref_ptr<PerspectiveCamera>& camera_left,
                 const atcg::ref_ptr<PerspectiveCamera>& camera_right);

    /**
     * @brief Gets called every frame
     *
     * @param delta_time Time since last frame
     */
    virtual bool onUpdate(float delta_time) override;

    /**
     * @brief Handles events
     *
     * @param e The event
     */
    virtual void onEvent(Event* e) override;

    /**
     * @brief Get the camera of the left eye
     *
     * @return The camera
     */
    ATCG_INLINE atcg::ref_ptr<atcg::PerspectiveCamera> getCameraLeft() const { return _cam_left; }

    /**
     * @brief Get the camera of the right eye
     *
     * @return The camera
     */
    ATCG_INLINE atcg::ref_ptr<atcg::PerspectiveCamera> getCameraRight() const { return _cam_right; }

    /**
     * @brief Get the position of the controller if the trigger is pressed for movement
     *
     * @return The Controller position
     */
    ATCG_INLINE glm::vec3 getControllerPosition() const { return _controller_position; }

    /**
     * @brief Get the direction of the controller if the trigger is pressed for movement
     *
     * @return The Controller direction
     */
    ATCG_INLINE glm::vec3 getControllerDirection() const { return _controller_direction; }

    /**
     * @brief Get the position of the intersection of the controller with the floor plane if the trigger is pressed for
     * movement
     *
     * @return The intersection
     */
    ATCG_INLINE glm::vec3 getControllerIntersection() const { return _controller_intersection; }

    /**
     * @brief Check if we are currently trying to move
     *
     * @return True if the trigger is pressed and we try to move
     */
    ATCG_INLINE bool inMovement() const { return _trigger_pressed; }

private:
    bool onWindowResize(WindowResizeEvent* event);
    bool onKeyPressed(KeyPressedEvent* event);
    bool onKeyReleased(KeyReleasedEvent* event);
    bool onVRButtonPressed(VRButtonPressedEvent* event);
    bool onVRButtonReleased(VRButtonReleasedEvent* event);

    float _speed              = 1.0f;
    float _velocity_forward   = 0.0f;
    float _velocity_right     = 0.0f;
    float _velocity_threshold = 0.3f;
    float _max_velocity       = 3.0f;
    float _acceleration       = 5.0f;

    bool _pressed_W       = false;
    bool _pressed_A       = false;
    bool _pressed_S       = false;
    bool _pressed_D       = false;
    bool _trigger_pressed = false;
    bool _trigger_release = false;
    uint32_t _device_index;
    glm::vec3 _controller_position;
    glm::vec3 _controller_direction;
    glm::vec3 _controller_intersection;

    atcg::ref_ptr<atcg::PerspectiveCamera> _cam_left;
    atcg::ref_ptr<atcg::PerspectiveCamera> _cam_right;
};

}    // namespace atcg