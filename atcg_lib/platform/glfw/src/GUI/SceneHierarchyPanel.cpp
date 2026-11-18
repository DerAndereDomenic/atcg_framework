#include <GUI/SceneHierarchyPanel.h>

#include <Core/Application.h>
#include <Scene/RevisionStack.h>

#include <Scene/Components/NameComponent.h>
#include <Scene/Components/IDComponent.h>

#include <imgui.h>
#include <portable-file-dialogs.h>

#include <Scene/ComponentRegistry.h>
#include <Scene/Clipboard.h>
#include <Utils/Utils.h>

#define ATCG_CONCAT_ID(name, uuid) ((name + ("##" + uuid)).c_str())

namespace atcg
{
namespace GUI
{

void SceneHierarchyPanel::drawEntityNode(const atcg::ref_ptr<Scene>& scene, Entity entity)
{
    auto& tag        = entity.getComponent<NameComponent>().name();
    bool is_selected = scene->isSelected(entity);

    ImGuiTreeNodeFlags flags = (is_selected ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_Bullet;
    flags |= ImGuiTreeNodeFlags_SpanAvailWidth;
    bool opened = ImGui::TreeNodeEx((void*)(uint64_t)entity.getComponent<IDComponent>().ID(), flags, tag.c_str());

    // Selected entities have a gray background (ImGuiCol_Header), the active entity is additionally outlined in the same
    // color as in the viewport (shader/outline.fs) if the entity selection is the most recent selection
    if(_entity_selection_active && is_selected && entity == scene->getActiveEntity())
    {
        ImGui::GetWindowDrawList()->AddRect(ImGui::GetItemRectMin(),
                                            ImGui::GetItemRectMax(),
                                            ImGui::GetColorU32(ImVec4(1.0f, 0.69f, 0.0f, 1.0f)),
                                            ImGui::GetStyle().FrameRounding,
                                            0,
                                            2.0f);
    }

    // Assets can be dropped onto the entity. If the entity is selected, the asset is applied to the whole selection.
    AssetHandle dropped_asset;
    if(Utils::acceptAssetDrop(dropped_asset,
                              [entity](AssetHandle handle) { return Utils::canApplyAsset(entity, handle); }))
    {
        _dropped_asset      = dropped_asset;
        _asset_drop_targets = is_selected ? scene->getSelectedEntities() : std::vector<Entity> {entity};
    }

    if(ImGui::IsItemClicked())
    {
        if(ImGui::GetIO().KeyCtrl)
        {
            scene->toggleEntitySelection(entity);
        }
        else
        {
            selectEntity(entity);
        }
    }

    // If the entity is part of a multi selection, the whole selection is deleted
    std::vector<Entity> selected_entities = scene->getSelectedEntities();
    bool delete_selection                 = is_selected && selected_entities.size() > 1;

    bool entityDeleted = false;
    if(ImGui::BeginPopupContextItem())
    {
        // Copying does not change the scene, so it is done immediately (the selection if the entity is selected)
        if(ImGui::MenuItem(ATCG_CONCAT_ID(std::string("Copy"), _uuid), "Ctrl+C"))
        {
            Clipboard::copyEntities(is_selected ? selected_entities : std::vector<Entity> {entity});
        }

        std::string label =
            delete_selection ? "Delete " + std::to_string(selected_entities.size()) + " Entities" : "Delete Entity";
        if(ImGui::MenuItem(ATCG_CONCAT_ID(label, _uuid))) entityDeleted = true;

        ImGui::EndPopup();
    }

    if(opened)
    {
        ImGui::TreePop();
    }

    if(entityDeleted)
    {
        // The entities are removed in renderPanel after the iteration over all entities has finished
        _entities_to_delete = delete_selection ? selected_entities : std::vector<Entity> {entity};
    }
}

void SceneHierarchyPanel::drawComponents(const atcg::ref_ptr<Scene>& scene, Entity entity)
{
    std::string id = std::to_string(entity.getComponent<IDComponent>().ID());
    std::stringstream label;

    NameComponent& component = entity.getComponent<NameComponent>();
    const std::string& tag   = component.name();
    char buffer[256];
    memset(buffer, 0, sizeof(buffer));
    // ? strncpy_s not available in gcc. Is this unsafe?
    memcpy(buffer, tag.c_str(), sizeof(buffer));
    label << "##" << id;
    if(ImGui::InputText(label.str().c_str(), buffer, sizeof(buffer)))
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<NameComponent>>(scene, entity);
        entity.addOrReplaceComponent<NameComponent>(std::string(buffer));
        atcg::RevisionStack::endRecording();
    }

    ImGui::SameLine();
    ImGui::PushItemWidth(-1);

    if(ImGui::Button(ATCG_CONCAT_ID("Add Component", _uuid)))
    {
        ImGui::OpenPopup(ATCG_CONCAT_ID("AddComponent", _uuid));
    }

    if(ImGui::BeginPopup(ATCG_CONCAT_ID("AddComponent", _uuid)))
    {
        ComponentRegistry::displayAddAllComponents(scene, entity);

        ImGui::EndPopup();
    }

    ImGui::PopItemWidth();

    ComponentRegistry::drawAllComponents(scene, entity);
}

void SceneHierarchyPanel::selectEntity(Entity entity)
{
    _focues_components = true;

    if(entity && entity.scene())
    {
        entity.scene()->selectEntity(entity);
    }
}

void SceneHierarchyPanel::renderPanel(const atcg::ref_ptr<Scene>& scene)
{
    // if(_selected_entity.scene() != scene.get())
    // {
    //     _selected_entity = {};
    // }
    auto selected_entity = scene->getActiveEntity();
    auto num_selected    = scene->getSelectedEntities().size();

    ImGui::Begin(ATCG_CONCAT_ID("Scene Hierarchy", _uuid));

    if(ImGui::IsMouseDown(0) && ImGui::IsWindowHovered() && !ImGui::IsAnyItemHovered() && !ImGui::IsAnyItemActive() &&
       !ImGui::GetIO().KeyCtrl)
    {
        scene->clearSelection();
    }

    // Pasting from the context menu is done after the iteration over all entities. The copy & paste shortcuts are
    // handled by the application, independent of the focused window.
    bool paste_entities = false;

    for(auto e: scene->getAllEntitiesWith<NameComponent>())
    {
        Entity entity(e, scene.get());
        if(entity.getComponent<NameComponent>().name() == "EditorCamera") continue;
        drawEntityNode(scene, entity);
    }

    if(!_entities_to_delete.empty())
    {
        // Removing an entity also removes it from the selection
        atcg::RevisionStack::startRecording<MultiEntityRevision<EntityRemovedRevision>>(scene, _entities_to_delete);
        for(auto entity: _entities_to_delete)
        {
            scene->removeEntity(entity);
        }
        atcg::RevisionStack::endRecording();
        _entities_to_delete.clear();

        selected_entity = scene->getActiveEntity();
        num_selected    = scene->getSelectedEntities().size();
    }

    if(!_asset_drop_targets.empty())
    {
        Utils::applyAssetToEntities(scene, _asset_drop_targets, _dropped_asset);
        _asset_drop_targets.clear();
        _dropped_asset = 0;
    }


    if(ImGui::BeginPopupContextWindow(0, ImGuiPopupFlags_MouseButtonRight | ImGuiPopupFlags_NoOpenOverExistingPopup))
    {
        if(ImGui::MenuItem(ATCG_CONCAT_ID("Create Empty Entity", _uuid)))
        {
            Entity entity = scene->createEntity("Empty Entity");
            atcg::RevisionStack::startRecording<EntityAddedRevision>(scene, entity);
            atcg::RevisionStack::endRecording();
            selectEntity(entity);
        }
        if(ImGui::MenuItem(ATCG_CONCAT_ID("Paste", _uuid), "Ctrl+V", false, Clipboard::hasEntities()))
        {
            paste_entities = true;
        }
        ImGui::EndPopup();
    }

    if(paste_entities)
    {
        // The pasted entities become the selection
        Clipboard::pasteEntities(scene);
        selected_entity = scene->getActiveEntity();
        num_selected    = scene->getSelectedEntities().size();
    }

    ImGui::End();

    ImGui::Begin(ATCG_CONCAT_ID("Properties", _uuid));
    ImGuiTabItemFlags flags = 0;
    if(_focues_components)
    {
        ImGui::SetWindowFocus();
        flags |= ImGuiTabItemFlags_SetSelected;
        _focues_components = false;
    }

    if(ImGui::BeginTabBar(ATCG_CONCAT_ID("TabBarComponents", _uuid)))
    {
        if(ImGui::BeginTabItem(ATCG_CONCAT_ID("Components", _uuid), (bool*)0, flags))
        {
            if(selected_entity)
            {
                if(num_selected > 1)
                {
                    ImGui::TextDisabled("%d entities selected (showing active)", (int)num_selected);
                }
                drawComponents(scene, selected_entity);
            }
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
}
}    // namespace GUI
}    // namespace atcg