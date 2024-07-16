#pragma once

#include <Core/Memory.h>
#include <Scene/Entity.h>
#include <Renderer/PerspectiveCamera.h>
#include <Scene/ComponentGUIHandler.h>

namespace atcg
{
class Scene;
class Framebuffer;

/**
 * @brief A Scene hierarchy panel
 *
 * @tparam GUIHandler The class that handles gui drawing. Default = atcg::ComponentGUIHandler
 */
template<typename GUIHandler = ComponentGUIHandler>
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
     *
     * @tparam CustomComponents... optional custom components that should be rendered.
     *
     * To support custom component rendering, a custom GUIHandler has to be provided.
     */
    template<typename... CustomComponents>
    void renderPanel();

    /**
     * @brief Set an entity as selected
     *
     * @param entity The entity
     */
    ATCG_INLINE void selectEntity(Entity entity);

    /**
     * @brief Get the currently selected entity
     *
     * @return The selected entity
     */
    ATCG_INLINE Entity getSelectedEntity() const { return _selected_entity; }

    /**
     * @brief Set a custom gui handler
     *
     * @param gui_handler The gui handler instance
     */
    ATCG_INLINE void setGuiHandler(const atcg::ref_ptr<GUIHandler>& gui_handler) { _gui_handler = gui_handler; }

private:
    void drawEntityNode(Entity entity);

    template<typename... Components>
    void drawComponents(Entity entity);

    void drawSceneProperties();
    Entity _selected_entity;
    atcg::ref_ptr<Scene> _scene;
    atcg::ref_ptr<GUIHandler> _gui_handler;

    bool _focues_components = false;
};
}    // namespace atcg


#include "../../src/Scene/SceneHierarchyPanelDetails.h"