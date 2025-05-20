#pragma once

#include <Core/Memory.h>
#include <Scene/Scene.h>
#include <Renderer/Camera.h>
#include <Renderer/Shader.h>

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

    /**
     * @brief Render an entity
     *
     * @param entity The entity to render
     * @param camera The camera
     */
    void render(Entity entity, const atcg::ref_ptr<Camera>& camera = {});

    /**
     * @brief Draw a specific component of an entity
     *
     * @tparam Component The component to draw. Must be ones of the RenderComponents in Components.h
     * @param entity The entity to draw
     * @param camera The camera
     * @param shader An optional shader. The shader (if not nullptr) overrides what is stored in the component AND what
     * is internally used. This means if you want to use a custom shader for a PointSphereRenderer component, for
     * example, you have to make sure that it is compatible with this rendering mode.
     */
    template<typename Component>
    void renderComponent(Entity entity,
                         const atcg::ref_ptr<Camera>& camera       = {},
                         const atcg::ref_ptr<atcg::Shader>& shader = nullptr);

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg