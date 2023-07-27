#pragma once

#include <Core/Memory.h>
#include <Scene/Scene.h>

namespace atcg
{
/**
 * @brief A Scene hierarchy panel
 */
class SceneHierarchyPanel
{
public:
    /**
     * @brief Default constructor
     */
    SceneHierarchyPanel() = default;

    /**
     * @brief Constructor
     *
     * @param scene The scene
     */
    SceneHierarchyPanel(const atcg::ref_ptr<Scene>& scene);

    /**
     * @brief Should be called in onImGuiRender.
     * Renders the panel
     */
    void renderPanel() const;

private:
    atcg::ref_ptr<Scene> _scene;
};
}    // namespace atcg