#pragma once

#include <Core/API.h>
#include <Core/Memory.h>
#include <Scene/Entity.h>
#include <Renderer/PerspectiveCamera.h>
#include <Scene/ComponentGUIHandler.h>

namespace atcg
{
class Scene;
class Framebuffer;

namespace GUI
{
/**
 * @brief A Scene hierarchy panel
 */
class ATCG_API SceneHierarchyPanel
{
public:
    /**
     * @brief Default constructor
     */
    SceneHierarchyPanel(const std::string uuid = "Main") : _uuid(uuid) {};

    /**
     * @brief Should be called in onImGuiRender.
     * Renders the panel
     */
    void renderPanel(const atcg::ref_ptr<Scene>& scene);

    /**
     * @brief Set an entity as selected
     *
     * @param entity The entity
     */
    void selectEntity(Entity entity);

    /**
     * @brief Get the currently selected entity
     *
     * @return The selected entity
     */
    ATCG_INLINE Entity getSelectedEntity() const { return _selected_entity; }

private:
    void drawEntityNode(const atcg::ref_ptr<Scene>& scene, Entity entity);

    void drawComponents(const atcg::ref_ptr<Scene>& scene, Entity entity);

    Entity _selected_entity;

    bool _focues_components = false;

    std::string _uuid;
};
}    // namespace GUI
}    // namespace atcg