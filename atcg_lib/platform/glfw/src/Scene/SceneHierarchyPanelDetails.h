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

ATCG_INLINE void SceneHierarchyPanel::drawSceneProperties()
{
    float content_scale                    = atcg::Application::get()->getWindow()->getContentScale();
    const ImGuiTreeNodeFlags treeNodeFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed |
                                             ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap |
                                             ImGuiTreeNodeFlags_FramePadding;

    bool open = ImGui::TreeNodeEx((void*)typeid(atcg::Scene).hash_code(), treeNodeFlags, "Skybox");

    ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();

    if(open)
    {
        if(_scene->hasSkybox())
        {
            ImGui::Image((ImTextureID)_scene->getSkyboxTexture()->getID(),
                         ImVec2(content_scale * 128, content_scale * 64),
                         ImVec2 {0, 1},
                         ImVec2 {1, 0});
            if(ImGui::Button("Remove skybox##skybox"))
            {
                _scene->removeSkybox();
            }
        }
        else
        {
            glm::vec4 clear_color = Renderer::getClearColor();
            if(ImGui::ColorEdit4("Background color#skybox", glm::value_ptr(clear_color)))
            {
                Renderer::setClearColor(clear_color);
            }

            if(ImGui::Button("Add Skybox..."))
            {
                auto f     = pfd::open_file("Choose files to read",
                                        pfd::path::home(),
                                            {"All Files",
                                             "*",
                                             "PNG Files (.png)",
                                             "*.png",
                                             "JPG Files (.jpg, .jpeg)",
                                             "*jpg, *jpeg",
                                             "BMP Files (.bmp)",
                                             "*.bmp",
                                             "HDR Files (.hdr)",
                                             "*.hdr"},
                                        pfd::opt::none);
                auto files = f.result();
                if(!files.empty())
                {
                    auto img = IO::imread(files[0]);
                    _scene->setSkybox(img);
                }
            }
        }
        ImGui::TreePop();
    }

    auto scene_camera = _scene->getCamera();
    if(scene_camera)
    {
        open = ImGui::TreeNodeEx((void*)typeid(atcg::Camera).hash_code(), treeNodeFlags, "Camera");

        contentRegionAvailable = ImGui::GetContentRegionAvail();

        if(open)
        {
            uint32_t width                    = atcg::Renderer::getFramebuffer()->width();
            uint32_t height                   = atcg::Renderer::getFramebuffer()->height();
            atcg::CameraIntrinsics intrinsics = scene_camera->getIntrinsics();
            glm::mat3 K                       = atcg::CameraUtils::convert_to_opencv(intrinsics, width, height);
            uint32_t id                       = (uint32_t)typeid(atcg::Camera).hash_code();
            bool updated                      = false;

            float fx = K[0][0];
            float fy = K[1][1];
            float cx = K[0][2];
            float cy = K[1][2];

            float f[2]      = {fx, fy};
            float c[2]      = {cx, cy};
            float offset[2] = {intrinsics.opticalCenter().x, intrinsics.opticalCenter().y};

            float aspect_ratio = intrinsics.aspectRatio();
            float fov          = intrinsics.FOV();

            std::stringstream label;
            label << "Aspect##" << id;
            if(ImGui::DragFloat(label.str().c_str(), &aspect_ratio, 0.05f, 0.1f, 5.0f))
            {
                intrinsics.setAspectRatio(aspect_ratio);
                updated = true;
            }

            ImGui::SameLine();
            if(ImGui::Button("Reset"))
            {
                intrinsics.setAspectRatio(float(width) / float(height));
                updated = true;
            }

            label.str(std::string());
            label << "FOV##" << id;
            if(ImGui::DragFloat(label.str().c_str(), &fov, 0.5f, 10.0f, 120.0f))
            {
                intrinsics.setFOV(fov);
                updated = true;
            }

            label.str(std::string());
            label << "Optical Center##" << id;
            if(ImGui::DragFloat2(label.str().c_str(), offset, 0.01f, -1.0f, 1.0f))
            {
                intrinsics.setOpticalCenter(glm::make_vec2(offset));
                updated = true;
            }

            ImGui::Separator();

            label.str(std::string());
            label << "Focal Length##" << id;
            if(ImGui::DragFloat2(label.str().c_str(), f, 0.5f, 1.0f, 4096.0f))
            {
                intrinsics = atcg::CameraUtils::convert_from_opencv(f[0],
                                                                    f[1],
                                                                    c[0],
                                                                    c[1],
                                                                    intrinsics.zNear(),
                                                                    intrinsics.zFar(),
                                                                    width,
                                                                    height);
                updated    = true;
            }

            label.str(std::string());
            label << "Principal Point##" << id;
            if(ImGui::DragFloat2(label.str().c_str(), c, 0.5f, 1.0f, 4096.0f))
            {
                intrinsics = atcg::CameraUtils::convert_from_opencv(f[0],
                                                                    f[1],
                                                                    c[0],
                                                                    c[1],
                                                                    intrinsics.zNear(),
                                                                    intrinsics.zFar(),
                                                                    width,
                                                                    height);
                updated    = true;
            }

            if(updated) scene_camera->setIntrinsics(intrinsics);

            ImGui::TreePop();
        }
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


        if(ImGui::BeginTabItem("Scene"))
        {
            drawSceneProperties();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
}
}    // namespace GUI
}    // namespace atcg