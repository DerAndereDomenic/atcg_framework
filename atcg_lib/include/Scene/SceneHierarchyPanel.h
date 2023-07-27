#pragma once

#include <Core/Memory.h>
#include <Scene/Scene.h>
#include <Scene/Entity.h>

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
    void renderPanel();

    /**
     * @brief Set an entity as selected
     *
     * @param entity The entity
     */
    inline void selectEntity(Entity entity) { _selected_entity = entity; }

    /**
     * @brief Get the currently selected entity
     *
     * @return The selected entity
     */
    inline Entity getSelectedEntity() const { return _selected_entity; }

private:
    void drawEntityNode(Entity entity);
    Entity _selected_entity;
    atcg::ref_ptr<Scene> _scene;
};
}    // namespace atcg