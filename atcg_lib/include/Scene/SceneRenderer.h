#pragma once

#include <Core/API.h>
#include <Core/Memory.h>
#include <Core/SystemRegistry.h>
#include <Renderer/Renderer.h>
#include <Scene/Scene.h>
#include <Renderer/RenderGraph.h>

namespace atcg
{
/**
 * @brief A system to render a scene using a render graph.
 */
class ATCG_API SceneRendererSystem
{
public:
    /**
     * @brief Constructor
     *
     * @param renderer The renderer to use for rendering the scene
     */
    SceneRendererSystem(RendererSystem* renderer);

    /**
     * @brief Destructor
     */
    ~SceneRendererSystem();

    /**
     * @brief Render a scene using the render graph.
     *
     * @param scene The scene to render
     * @param camera The camera to render the scene from
     * @param target_fbo The framebuffer to render the scene into
     * @param draw_cameras If true, the registered cameras in the scene are rendered as well
     */
    void render(const atcg::ref_ptr<Scene>& scene,
                const atcg::ref_ptr<Camera>& camera,
                const atcg::ref_ptr<Framebuffer>& target_fbo,
                bool draw_cameras = true);

    /**
     * @brief Get the render graph used by this renderer system
     *
     * @return The render graph used by this renderer system
     */
    atcg::ref_ptr<RenderGraph> getRenderGraph() const;

    /**
     * @brief Set the render graph used by this renderer system
     *
     * @param graph The render graph to use in this renderer system
     */
    void setRenderGraph(const atcg::ref_ptr<RenderGraph>& graph);

    /**
     * @brief Set the number of MSAA samples used in the forward pass. This requires recompiling the render graph.
     *
     * @param num_samples The number of MSAA samples to use in the forward pass
     */
    void setNumberMSAASamples(uint32_t num_samples);

    /**
     * @brief Set the resolution of the shadow maps used in the shadow pass. This requires recompiling the render graph.
     *
     * @param resolution The resolution (width and height) of the shadow maps to use in the shadow pass
     */
    void setShadowPassResolution(uint32_t resolution);

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

namespace SceneRenderer
{
/**
 * @brief Render a scene using the render graph.
 *
 * @param scene The scene to render
 * @param camera The camera to render the scene from
 * @param target_fbo The framebuffer to render the scene into
 * @param draw_cameras If true, the registered cameras in the scene are rendered as well
 */
ATCG_INLINE void render(const atcg::ref_ptr<Scene>& scene,
                        const atcg::ref_ptr<Camera>& camera,
                        const atcg::ref_ptr<Framebuffer>& target_fbo,
                        bool draw_cameras = true)
{
    SystemRegistry::instance()->getSystem<SceneRendererSystem>()->render(scene, camera, target_fbo, draw_cameras);
}

/**
 * @brief Get the render graph used by this renderer system
 *
 * @return The render graph used by this renderer system
 */
ATCG_INLINE atcg::ref_ptr<RenderGraph> getRenderGraph()
{
    return SystemRegistry::instance()->getSystem<SceneRendererSystem>()->getRenderGraph();
}

/**
 * @brief Set the render graph used by this renderer system
 *
 * @param graph The render graph to use in this renderer system
 */
ATCG_INLINE void setRenderGraph(atcg::ref_ptr<RenderGraph>& graph)
{
    SystemRegistry::instance()->getSystem<SceneRendererSystem>()->setRenderGraph(graph);
}

/**
 * @brief Set the number of MSAA samples used in the forward pass. This requires recompiling the render graph.
 *
 * @param num_samples The number of MSAA samples to use in the forward pass
 */
ATCG_INLINE void setNumberMSAASamples(uint32_t num_samples)
{
    SystemRegistry::instance()->getSystem<SceneRendererSystem>()->setNumberMSAASamples(num_samples);
}

/**
 * @brief Set the resolution of the shadow maps used in the shadow pass. This requires recompiling the render graph.
 *
 * @param resolution The resolution (width and height) of the shadow maps to use in the shadow pass
 */
ATCG_INLINE void setShadowPassResolution(uint32_t resolution)
{
    SystemRegistry::instance()->getSystem<SceneRendererSystem>()->setShadowPassResolution(resolution);
}

}    // namespace SceneRenderer

}    // namespace atcg