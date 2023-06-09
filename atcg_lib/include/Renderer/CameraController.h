#pragma once

#include <Renderer/PerspectiveCamera.h>
#include <Events/Event.h>
#include <Events/MouseEvent.h>
#include <Events/KeyEvent.h>
#include <Events/WindowEvent.h>

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
     * @brief Construct a new Camera Controller object
     *
     * @param aspect_ratio The aspect ratio of the camera
     */
    CameraController(const float& aspect_ratio);

    /**
     * @brief Gets called every frame
     *
     * @param delta_time Time since last frame
     */
    virtual void onUpdate(float delta_time) = 0;

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
    inline const atcg::ref_ptr<PerspectiveCamera>& getCamera() const { return _camera; }

protected:
    atcg::ref_ptr<PerspectiveCamera> _camera;
};

class FocusedController : public CameraController
{
public:
    /**
     * @brief Construct a new Focus Camera object
     *
     * @param aspect_ratio The aspect ratio of the camera
     */
    FocusedController(const float& aspect_ratio);

    /**
     * @brief Gets called every frame
     *
     * @param delta_time Time since last frame
     */
    virtual void onUpdate(float delta_time);

    /**
     * @brief Handles events
     *
     * @param e The event
     */
    virtual void onEvent(Event* e);

private:
    bool onMouseZoom(MouseScrolledEvent* event);
    bool onWindowResize(WindowResizeEvent* event);
    bool onMouseMove(MouseMovedEvent* event);

    float _distance = 1.0f;
    float _zoom_speed = 0.25f;
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
     * @param aspect_ratio The aspect ratio of the camera
     */
    FirstPersonController(const float& aspect_ratio);

    /**
     * @brief Gets called every frame
     *
     * @param delta_time Time since last frame
     */
    virtual void onUpdate(float delta_time);

    /**
     * @brief Handles events
     *
     * @param e The event
     */
    virtual void onEvent(Event* e);

private:
    bool onWindowResize(WindowResizeEvent* event);
    bool onMouseMove(MouseMovedEvent* event);
    bool onKeyPressed(KeyPressedEvent* event);

    float _speed = 1.0f;
    float _velocity_forward = 0.0f;
    float _velocity_right = 0.0f;
    float _velocity_up = 0.0f;
    float _velocity_threshold = 0.005f;
    float _max_velocity = 0.1f;
    float _acceleration = 0.004f;
    float _rotation_speed = 0.005f;
    float _lastX = 0, _lastY = 0;
    float _currentX = 0, _currentY = 0;
};

}    // namespace atcg