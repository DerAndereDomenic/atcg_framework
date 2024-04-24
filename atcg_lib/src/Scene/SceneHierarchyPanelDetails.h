#pragma once

#include <imgui.h>

namespace atcg
{

namespace detail
{
template<typename T>
inline void drawComponent(Entity entity, const atcg::ref_ptr<ComponentGUIHandler>& gui_handler)
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
            gui_handler->draw_component<T>(entity, component);
            ImGui::TreePop();
        }

        if(removeComponent) entity.removeComponent<T>();
    }
}

template<typename T>
inline void displayAddComponentEntry(Entity entity)
{
    if(!entity.hasComponent<T>())
    {
        if(ImGui::MenuItem(T::toString()))
        {
            entity.addComponent<T>();
            ImGui::CloseCurrentPopup();
        }
    }
}

template<>
inline void displayAddComponentEntry<CameraComponent>(Entity entity)
{
    if(!entity.hasComponent<CameraComponent>())
    {
        if(ImGui::MenuItem(CameraComponent::toString()))
        {
            auto& camera_component = entity.addComponent<CameraComponent>(atcg::make_ref<PerspectiveCamera>(1.0f));
            if(entity.hasComponent<TransformComponent>())
            {
                atcg::ref_ptr<PerspectiveCamera> cam = camera_component.camera;
                cam->setView(glm::inverse(entity.getComponent<TransformComponent>().getModel()));
            }
            ImGui::CloseCurrentPopup();
        }
    }
}

}    // namespace detail

template<typename... Components>
inline void SceneHierarchyPanel::drawComponents(Entity entity)
{
    std::string id = std::to_string(entity.getComponent<IDComponent>().ID);
    std::stringstream label;

    float content_scale = atcg::Application::get()->getWindow()->getContentScale();

    NameComponent& component = entity.getComponent<NameComponent>();
    std::string& tag         = component.name;
    char buffer[256];
    memset(buffer, 0, sizeof(buffer));
    // ? strncpy_s not available in gcc. Is this unsafe?
    memcpy(buffer, tag.c_str(), sizeof(buffer));
    label << "##" << id;
    if(ImGui::InputText(label.str().c_str(), buffer, sizeof(buffer)))
    {
        tag = std::string(buffer);
    }

    ImGui::SameLine();
    ImGui::PushItemWidth(-1);

    if(ImGui::Button("Add Component"))
    {
        ImGui::OpenPopup("AddComponent");
    }

    if(ImGui::BeginPopup("AddComponent"))
    {
        (detail::displayAddComponentEntry<Components>(entity), ...);

        ImGui::EndPopup();
    }

    ImGui::PopItemWidth();

    (detail::drawComponent<Components>(entity, _gui_handler), ...);
}

template<typename... CustomComponents>
inline void SceneHierarchyPanel::renderPanel()
{
    ImGui::Begin("Scene Hierarchy");

    for(auto e: _scene->getAllEntitiesWith<NameComponent>())
    {
        Entity entity(e, _scene.get());
        if(entity.getComponent<NameComponent>().name == "EditorCamera") continue;
        drawEntityNode(entity);
    }

    if(ImGui::IsMouseDown(0) && ImGui::IsWindowHovered())
    {
        _selected_entity = {};
    }

    if(ImGui::BeginPopupContextWindow(0, 1))
    {
        if(ImGui::MenuItem("Create Empty Entity"))
        {
            Entity entity    = _scene->createEntity("Empty Entity");
            _selected_entity = entity;
        }
        ImGui::EndPopup();
    }

    ImGui::End();

    ImGui::Begin("Properties");

    if(ImGui::BeginTabBar("TabBarComponents"))
    {
        if(ImGui::BeginTabItem("Components"))
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
                               CustomComponents...>(_selected_entity);
            }
            ImGui::EndTabItem();
        }


        if(ImGui::BeginTabItem("Scene"))
        {
            drawSceneProperties();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
}
}