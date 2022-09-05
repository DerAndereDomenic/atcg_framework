#pragma once

#include <Renderer/PerspectiveCamera.h>
#include <Events/Event.h>
#include <Events/MouseEvent.h>
#include <Events/WindowEvent.h>

#include <memory>

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
        void onUpdate(float delta_time);

        /**
         * @brief Handles events
         * 
         * @param e The event
         */
        void onEvent(Event& e);

        /**
         * @brief Get the Camera object
         * 
         * @return const std::unique_ptr<Camera>& The camera
         */
        inline const std::shared_ptr<PerspectiveCamera>& getCamera() const {return _camera;}

    private:
        bool onMouseZoom(MouseScrolledEvent& event);
        bool onWindowResize(WindowResizeEvent& event);
        bool onMouseMove(MouseMovedEvent& event);

        // Adjustable only through here for now
        struct CameraParameters
        {
            float zoom_speed = 0.25f;
            float rotation_speed = 0.2f;
        };

        float _distance;
        CameraParameters _parameters;
        std::shared_ptr<PerspectiveCamera> _camera;
        float _lastX = 0, _lastY = 0;
        float _currentX = 0, _currentY = 0;
    };
}