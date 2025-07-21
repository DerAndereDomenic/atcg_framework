#pragma once

#include <Scene/RevisionStack.h>

#include <imgui.h>
#include <portable-file-dialogs.h>

namespace atcg
{
namespace GUI
{
namespace detail
{
template<typename T>
ATCG_INLINE void drawComponent(const atcg::ref_ptr<Scene>& scene, Entity entity)
{
    const ImGuiTreeNodeFlags treeNodeFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed |
                                             ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap |
                                             ImGuiTreeNodeFlags_FramePadding;

    if(entity.hasComponent<T>())
    {
        auto& component = entity.getComponent<T>();
        bool open       = ImGui::TreeNodeEx((void*)typeid(T).hash_code(), treeNodeFlags, T::toString());

        ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();
        ImGui::SameLine(contentRegionAvailable.x);
        if(ImGui::Button("+"))
        {
            ImGui::OpenPopup("ComponentSettings");
        }

        bool removeComponent = false;
        if(ImGui::BeginPopup("ComponentSettings"))
        {
            if(ImGui::MenuItem("Remove component")) removeComponent = true;

            ImGui::EndPopup();
        }

        if(open)
        {
            ComponentGUIRenderer<T>().draw_component(scene, entity, component);
            ImGui::TreePop();
        }

        if(removeComponent)
        {
            atcg::RevisionStack::startRecording<ComponentRemovedRevision<T>>(scene, entity);
            entity.removeComponent<T>();
            atcg::RevisionStack::endRecording();
        }
    }
}

template<typename T>
ATCG_INLINE void displayAddComponentEntry(const atcg::ref_ptr<atcg::Scene>& scene, Entity entity)
{
    if(!entity.hasComponent<T>())
    {
        if(ImGui::MenuItem(T::toString()))
        {
            atcg::RevisionStack::startRecording<ComponentAddedRevision<T>>(scene, entity);
            entity.addComponent<T>();
            ImGui::CloseCurrentPopup();
            atcg::RevisionStack::endRecording();
        }
    }
}

template<>
ATCG_INLINE void displayAddComponentEntry<CameraComponent>(const atcg::ref_ptr<atcg::Scene>& scene, Entity entity)
{
    if(!entity.hasComponent<CameraComponent>())
    {
        if(ImGui::MenuItem(CameraComponent::toString()))
        {
            atcg::RevisionStack::startRecording<ComponentAddedRevision<CameraComponent>>(scene, entity);
            auto& camera_component = entity.addComponent<CameraComponent>(atcg::make_ref<PerspectiveCamera>());
            if(entity.hasComponent<TransformComponent>())
            {
                atcg::ref_ptr<PerspectiveCamera> cam =
                    std::dynamic_pointer_cast<PerspectiveCamera>(camera_component.camera);
                cam->setView(glm::inverse(entity.getComponent<TransformComponent>().getModel()));
            }
            ImGui::CloseCurrentPopup();
            atcg::RevisionStack::endRecording();
        }
    }
}

}    // namespace detail

ATCG_INLINE SceneHierarchyPanel::SceneHierarchyPanel(const atcg::ref_ptr<Scene>& scene) : _scene(scene) {}

ATCG_INLINE void SceneHierarchyPanel::drawEntityNode(Entity entity)
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
        if(ImGui::MenuItem("Delete Entity")) entityDeleted = true;

        ImGui::EndPopup();
    }

    if(opened)
    {
        ImGui::TreePop();
    }

    if(entityDeleted)
    {
        atcg::RevisionStack::startRecording<EntityRemovedRevision>(_scene, entity);
        if(_selected_entity == entity) selectEntity({});
        _scene->removeEntity(entity);
        atcg::RevisionStack::endRecording();
    }
}

template<typename... Components>
ATCG_INLINE void SceneHierarchyPanel::drawComponents(Entity entity)
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
        atcg::RevisionStack::startRecording<ComponentEditedRevision<NameComponent>>(_scene, entity);
        entity.addOrReplaceComponent<NameComponent>(std::string(buffer));
        atcg::RevisionStack::endRecording();
    }

    ImGui::SameLine();
    ImGui::PushItemWidth(-1);

    if(ImGui::Button("Add Component"))
    {
        ImGui::OpenPopup("AddComponent");
    }

    if(ImGui::BeginPopup("AddComponent"))
    {
        (detail::displayAddComponentEntry<Components>(_scene, entity), ...);

        ImGui::EndPopup();
    }

    ImGui::PopItemWidth();

    (detail::drawComponent<Components>(_scene, entity), ...);
}

ATCG_INLINE void SceneHierarchyPanel::selectEntity(Entity entity)
{
    _selected_entity   = entity;
    _focues_components = true;
}

template<typename... CustomComponents>
ATCG_INLINE void SceneHierarchyPanel::renderPanel()
{
    ImGui::Begin("Scene Hierarchy");

    if(ImGui::IsMouseDown(0) && ImGui::IsWindowHovered() && !ImGui::IsAnyItemHovered() && !ImGui::IsAnyItemActive())
    {
        selectEntity({});
    }

    for(auto e: _scene->getAllEntitiesWith<NameComponent>())
    {
        Entity entity(e, _scene.get());
        if(entity.getComponent<NameComponent>().name() == "EditorCamera") continue;
        drawEntityNode(entity);
    }


    if(ImGui::BeginPopupContextWindow(0, ImGuiPopupFlags_MouseButtonRight | ImGuiPopupFlags_NoOpenOverExistingPopup))
    {
        if(ImGui::MenuItem("Create Empty Entity"))
        {
            Entity entity = _scene->createEntity("Empty Entity");
            atcg::RevisionStack::startRecording<EntityAddedRevision>(_scene, entity);
            atcg::RevisionStack::endRecording();
            selectEntity(entity);
        }
        ImGui::EndPopup();
    }

    ImGui::End();

    ImGui::Begin("Properties");
    ImGuiTabItemFlags flags = 0;
    if(_focues_components)
    {
        ImGui::SetWindowFocus();
        flags |= ImGuiTabItemFlags_SetSelected;
        _focues_components = false;
    }

    if(ImGui::BeginTabBar("TabBarComponents"))
    {
        if(ImGui::BeginTabItem("Components", (bool*)0, flags))
        {
            if(_selected_entity)
            {
                drawComponents<CameraComponent,
                               TransformComponent,
                               GeometryComponent,
                               MeshRenderComponent,
                               PointRenderComponent,
                               PointSphereRenderComponent,
                               EdgeRenderComponent,
                               EdgeCylinderRenderComponent,
                               InstanceRenderComponent,
                               PointLightComponent,
                               ScriptComponent,
                               CustomComponents...>(_selected_entity);
            }
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
}
}    // namespace GUI
}    // namespace atcg