#pragma once

#include <Core/API.h>
#include <Core/Memory.h>
#include <Asset/Asset.h>
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

    /**
     * @brief Set whether the entity selection is the most recent selection of the application.
     * Only then the active entity is outlined. If something else was selected afterwards (e.g. an asset in the
     * AssetPanel, see AssetPanel::isAssetSelectionActive), the selected entities are only highlighted.
     *
     * @param active If the entity selection is the most recent selection (default: true)
     */
    ATCG_INLINE void setEntitySelectionActive(bool active) { _entity_selection_active = active; }

private:
    void selectEntity(Entity entity);

    // Creates and selects an entity. The components are added by the callback and recorded as one undo step.
    void createEntity(const atcg::ref_ptr<Scene>& scene,
                      const std::string& name,
                      const std::function<void(Entity)>& add_components);

    void drawEntityNode(const atcg::ref_ptr<Scene>& scene, Entity entity);

    void drawComponents(const atcg::ref_ptr<Scene>& scene, Entity entity);

    bool _focues_components = false;

    bool _entity_selection_active = true;

    // Entities are deleted after iterating over the scene's entities
    std::vector<Entity> _entities_to_delete;

    // Assets dropped onto entities are applied after iterating over the scene's entities (components may be added)
    AssetHandle _dropped_asset = 0;
    std::vector<Entity> _asset_drop_targets;

    std::string _uuid = "Main";
};
}    // namespace GUI
}    // namespace atcg