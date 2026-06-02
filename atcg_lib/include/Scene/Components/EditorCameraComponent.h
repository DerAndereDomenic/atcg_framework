#pragma once

#include <Renderer/Camera.h>
#include <Scene/Components/CameraComponent.h>

namespace atcg
{
struct EditorCameraComponent : public CameraComponent
{
    EditorCameraComponent() = default;
    EditorCameraComponent(const atcg::ref_ptr<Camera>& camera) : CameraComponent(camera) {}
};
}    // namespace atcg