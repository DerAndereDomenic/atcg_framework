#include <Scene/SceneHierarchyPanel.h>

#include <imgui.h>
#include <Scene/Components.h>
#include <portable-file-dialogs.h>
#include <DataStructure/TorchUtils.h>
#include <Core/Application.h>

namespace atcg
{

namespace detail
{
template<typename T>
void drawComponent(const std::string& name, Entity entity, const atcg::ref_ptr<ComponentGUIHandler>& gui_handler)
{
    const ImGuiTreeNodeFlags treeNodeFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed |
                                             ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap |
                                             ImGuiTreeNodeFlags_FramePadding;

    if(entity.hasComponent<T>())
    {
        auto& component = entity.getComponent<T>();
        bool open       = ImGui::TreeNodeEx((void*)typeid(T).hash_code(), treeNodeFlags, name.c_str());

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
void displayAddComponentEntry(const std::string& name, Entity entity)
{
    if(!entity.hasComponent<T>())
    {
        if(ImGui::MenuItem(name.c_str()))
        {
            entity.addComponent<T>();
            ImGui::CloseCurrentPopup();
        }
    }
}

template<>
void displayAddComponentEntry<CameraComponent>(const std::string& name, Entity entity)
{
    if(!entity.hasComponent<CameraComponent>())
    {
        if(ImGui::MenuItem(name.c_str()))
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

void SceneHierarchyPanel::drawEntityNode(Entity entity)
{
    auto& tag = entity.getComponent<NameComponent>().name;

    ImGuiTreeNodeFlags flags =
        ((_selected_entity && _selected_entity.getComponent<IDComponent>().ID == entity.getComponent<IDComponent>().ID)
             ? ImGuiTreeNodeFlags_Selected
             : 0) |
        ImGuiTreeNodeFlags_Bullet;
    flags |= ImGuiTreeNodeFlags_SpanAvailWidth;
    bool opened = ImGui::TreeNodeEx((void*)(uint64_t)entity.getComponent<IDComponent>().ID, flags, tag.c_str());
    if(ImGui::IsItemClicked())
    {
        _selected_entity = entity;
    }

    // bool entityDeleted = false;
    // if(ImGui::BeginPopupContextItem())
    // {
    //     if(ImGui::MenuItem("Delete Entity")) entityDeleted = true;

    //     ImGui::EndPopup();
    // }

    if(opened)
    {
        ImGui::TreePop();
    }

    // if(entityDeleted)
    // {
    //     m_Context->DestroyEntity(entity);
    //     if(m_SelectionContext == entity) m_SelectionContext = {};
    // }
}

void SceneHierarchyPanel::drawComponents(Entity entity)
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
        detail::displayAddComponentEntry<TransformComponent>("Transform", entity);
        detail::displayAddComponentEntry<GeometryComponent>("Geometry", entity);
        detail::displayAddComponentEntry<MeshRenderComponent>("Mesh Renderer", entity);
        detail::displayAddComponentEntry<PointRenderComponent>("Point Renderer", entity);
        detail::displayAddComponentEntry<PointSphereRenderComponent>("Point Sphere Renderer", entity);
        detail::displayAddComponentEntry<EdgeRenderComponent>("Edge Renderer", entity);
        detail::displayAddComponentEntry<EdgeCylinderRenderComponent>("Edge Cylinder Renderer", entity);
        detail::displayAddComponentEntry<CameraComponent>("Camera Component", entity);
        ImGui::EndPopup();
    }

    ImGui::PopItemWidth();

    detail::drawComponent<CameraComponent>("Camera View", entity, _gui_handler);
    detail::drawComponent<TransformComponent>("Transform", entity, _gui_handler);
    detail::drawComponent<GeometryComponent>("Geometry", entity, _gui_handler);
    detail::drawComponent<MeshRenderComponent>("Mesh Renderer", entity, _gui_handler);
    detail::drawComponent<PointRenderComponent>("Point Renderer", entity, _gui_handler);
    detail::drawComponent<PointSphereRenderComponent>("Point Sphere Renderer", entity, _gui_handler);
    detail::drawComponent<EdgeRenderComponent>("Edge Renderer", entity, _gui_handler);
    detail::drawComponent<EdgeCylinderRenderComponent>("Edge Cylinder Renderer", entity, _gui_handler);
    detail::drawComponent<InstanceRenderComponent>("Instance Renderer", entity, _gui_handler);
}

void SceneHierarchyPanel::drawSceneProperties()
{
    float content_scale                    = atcg::Application::get()->getWindow()->getContentScale();
    const ImGuiTreeNodeFlags treeNodeFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed |
                                             ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap |
                                             ImGuiTreeNodeFlags_FramePadding;

    bool open = ImGui::TreeNodeEx((void*)typeid(atcg::Scene).hash_code(), treeNodeFlags, "Skybox");

    ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();

    if(open)
    {
        if(Renderer::hasSkybox())
        {
            ImGui::Image((void*)(uint64_t)Renderer::getSkyboxTexture()->getID(),
                         ImVec2(content_scale * 128, content_scale * 64),
                         ImVec2 {0, 1},
                         ImVec2 {1, 0});
            if(ImGui::Button("Remove skybox##skybox"))
            {
                Renderer::removeSkybox();
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
                    Renderer::setSkybox(img);
                }
            }
        }
        ImGui::TreePop();
    }
}

SceneHierarchyPanel::SceneHierarchyPanel(const atcg::ref_ptr<Scene>& scene)
    : _scene(scene),
      _gui_handler(atcg::make_ref<atcg::ComponentGUIHandler>(scene))
{
}

void SceneHierarchyPanel::renderPanel()
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
                drawComponents(_selected_entity);
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
}    // namespace atcg