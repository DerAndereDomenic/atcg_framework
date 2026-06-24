#pragma once

// #include <Renderer/Renderer.h>
// #include <Scene/Scene.h>
// #include <Renderer/Camera.h>
#include <Core/API.h>


namespace atcg
{

class RendererSystem;
class Scene;
class Camera;

struct ATCG_API RenderContext
{
    RendererSystem* renderer;
    atcg::ref_ptr<Scene> scene;
    atcg::ref_ptr<Camera> camera;
    bool draw_cameras = true;
};
}    // namespace atcg