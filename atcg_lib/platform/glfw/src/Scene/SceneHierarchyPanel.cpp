#include <Scene/SceneHierarchyPanel.h>

#include <Core/Application.h>
#include <Scene/RevisionStack.h>

#include <imgui.h>
#include <portable-file-dialogs.h>

#include <Scene/ComponentRegistry.h>

#define ATCG_CONCAT_ID(name, uuid) ((name + ("##" + uuid)).c_str())

namespace atcg
{
namespace GUI
{

void SceneHierarchyPanel::drawEntityNode(const atcg::ref_ptr<Scene>& scene, Entity entity)
{
    auto& tag = entity.getComponent<NameComponent>().name();

    ImGuiTreeNodeFlags flags = ((_selected_entity && _selected_entity.getComponent<IDComponent>().ID() ==
                                                         entity.getComponent<IDComponent>().ID())
                                    ? ImGuiTreeNodeFlags_Selected
                                    : 0) |
                               ImGuiTreeNodeFlags_Bullet;
    flags |= ImGuiTreeNodeFlags_SpanAvailWidth;
    bool opened = ImGui::TreeNodeEx((void*)(uint64_t)entity.getComponent<IDComponent>().ID(), flags, tag.c_str());
    if(ImGui::IsItemClicked())
    {
        selectEntity(entity);
    }

    bool entityDeleted = false;
    if(ImGui::BeginPopupContextItem())
    {
        if(ImGui::MenuItem(ATCG_CONCAT_ID("Delete Entity", _uuid))) entityDeleted = true;

        ImGui::EndPopup();
    }

    if(opened)
    {
        ImGui::TreePop();
    }

    if(entityDeleted)
    {
        atcg::RevisionStack::startRecording<EntityRemovedRevision>(scene, entity);
        if(_selected_entity == entity) selectEntity({});
        scene->removeEntity(entity);
        atcg::RevisionStack::endRecording();
    }
}

void SceneHierarchyPanel::drawComponents(const atcg::ref_ptr<Scene>& scene, Entity entity)
{
    std::string id = std::to_string(entity.getComponent<IDComponent>().ID());
    std::stringstream label;

    float content_scale = atcg::Application::get()->getWindow()->getContentScale();

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
    _selected_entity   = entity;
    _focues_components = true;
}

void SceneHierarchyPanel::renderPanel(const atcg::ref_ptr<Scene>& scene)
{
    if(_selected_entity.scene() != scene.get())
    {
        _selected_entity = {};
    }

    ImGui::Begin(ATCG_CONCAT_ID("Scene Hierarchy", _uuid));

    if(ImGui::IsMouseDown(0) && ImGui::IsWindowHovered() && !ImGui::IsAnyItemHovered() && !ImGui::IsAnyItemActive())
    {
        selectEntity({});
    }

    for(auto e: scene->getAllEntitiesWith<NameComponent>())
    {
        Entity entity(e, scene.get());
        if(entity.getComponent<NameComponent>().name() == "EditorCamera") continue;
        drawEntityNode(scene, entity);
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
        ImGui::EndPopup();
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
            if(_selected_entity)
            {
                drawComponents(scene, _selected_entity);
            }
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
}
}    // namespace GUI
}    // namespace atcg