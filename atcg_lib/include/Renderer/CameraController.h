#pragma once

#include <Renderer/Camera.h>
#include <Events/Event.h>

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
        std::unique_ptr<Camera> _camera;
    };
}