#pragma once

#include <Core/Memory.h>
#include <Scene/Scene.h>
#include <Renderer/Camera.h>

namespace atcg
{
class SceneRenderer
{
public:
    SceneRenderer();

    SceneRenderer(const atcg::ref_ptr<atcg::Scene>& scene);

    ~SceneRenderer();

    void setScene(const atcg::ref_ptr<atcg::Scene>& scene);

    void render(const atcg::ref_ptr<Camera>& camera);

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg