#pragma once

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
    void selectEntity(Entity entity);

    /**
     * @brief Get the currently selected entity
     *
     * @return The selected entity
     */
    ATCG_INLINE Entity getSelectedEntity() const { return _selected_entity; }

private:
    void drawEntityNode(Entity entity);

    template<typename... Components>
    void drawComponents(Entity entity);

    void drawSceneProperties();
    Entity _selected_entity;
    atcg::ref_ptr<Scene> _scene;

    bool _focues_components = false;
};
}    // namespace GUI
}    // namespace atcg

#ifndef ATCG_HEADLESS
    #include "../../platform/glfw/src/Scene/SceneHierarchyPanelDetails.h"
#else
    #include "../../platform/headless/src/Scene/SceneHierarchyPanelDetails.h"
#endif