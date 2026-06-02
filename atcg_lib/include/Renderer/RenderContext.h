#pragma once

// #include <Renderer/Renderer.h>
// #include <Scene/Scene.h>
// #include <Renderer/Camera.h>


namespace atcg
{

class RendererSystem;
class Scene;
class Camera;

struct RenderContext
{
    RendererSystem* renderer;
    atcg::ref_ptr<Scene> scene;
    atcg::ref_ptr<Camera> camera;
    bool draw_cameras = true;
};
}    // namespace atcg