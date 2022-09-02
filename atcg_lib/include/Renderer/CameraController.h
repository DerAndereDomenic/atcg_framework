#pragma once

#include <Renderer/Camera.h>
#include <Events/Event.h>
#include <Events/MouseEvent.h>
#include <Events/WindowEvent.h>

#include <memory>

namespace atcg
{
    class CameraController
    {
    public:
        CameraController(const float& aspect_ratio);

        void onUpdate(float delta_time);

        void onEvent(Event& e);

        inline const std::unique_ptr<Camera>& getCamera() const {return _camera;}

    private:
        bool onMouseZoom(MouseScrolledEvent& event);
        bool onWindowResize(WindowResizeEvent& event);

        // Adjustable only through here for now
        struct CameraParameters
        {
            float zoom_speed = 0.25f;
        };

        float _distance;
        CameraParameters _parameters;
        std::unique_ptr<Camera> _camera;
    };
}