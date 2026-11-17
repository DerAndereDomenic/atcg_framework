#pragma once

#include <Core/API.h>
#include <Core/Memory.h>
#include <Scene/Entity.h>
#include <Renderer/PerspectiveCamera.h>
#include <GUI/ComponentGUIHandler.h>

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
    SceneHierarchyPanel(const std::string& uuid = "Main") : _uuid(uuid) {};

    /**
     * @brief Should be called in onImGuiRender.
     * Renders the panel
     */
    void renderPanel(const atcg::ref_ptr<Scene>& scene);


private:
    void selectEntity(Entity entity);

    void drawEntityNode(const atcg::ref_ptr<Scene>& scene, Entity entity);

    void drawComponents(const atcg::ref_ptr<Scene>& scene, Entity entity);

    bool _focues_components = false;

    std::string _uuid = "Main";
};
}    // namespace GUI
}    // namespace atcg