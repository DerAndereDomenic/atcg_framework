#include <Scene/SceneHierarchyPanel.h>

#include <imgui.h>
#include <Scene/Components.h>

#include <stb_image_write.h>

namespace atcg
{

namespace detail
{
template<typename T, typename UIFunction>
void drawComponent(const std::string& name, Entity entity, UIFunction uiFunction)
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
        if(ImGui::Button("+")) { ImGui::OpenPopup("ComponentSettings"); }

        bool removeComponent = false;
        if(ImGui::BeginPopup("ComponentSettings"))
        {
            if(ImGui::MenuItem("Remove component")) removeComponent = true;

            ImGui::EndPopup();
        }

        if(open)
        {
            uiFunction(component);
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
    if(ImGui::IsItemClicked()) { _selected_entity = entity; }

    // bool entityDeleted = false;
    // if(ImGui::BeginPopupContextItem())
    // {
    //     if(ImGui::MenuItem("Delete Entity")) entityDeleted = true;

    //     ImGui::EndPopup();
    // }

    if(opened) { ImGui::TreePop(); }

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

    NameComponent& component = entity.getComponent<NameComponent>();
    std::string& tag         = component.name;
    char buffer[256];
    memset(buffer, 0, sizeof(buffer));
    // ? strncpy_s not available in gcc. Is this unsafe?
    memcpy(buffer, tag.c_str(), sizeof(buffer));
    label << "##" << id;
    if(ImGui::InputText(label.str().c_str(), buffer, sizeof(buffer))) { tag = std::string(buffer); }

    ImGui::SameLine();
    ImGui::PushItemWidth(-1);

    if(ImGui::Button("Add Component")) { ImGui::OpenPopup("AddComponent"); }

    if(ImGui::BeginPopup("AddComponent"))
    {
        detail::displayAddComponentEntry<TransformComponent>("Transform", entity);
        detail::displayAddComponentEntry<MeshRenderComponent>("Mesh Renderer", entity);
        detail::displayAddComponentEntry<PointRenderComponent>("Point Renderer", entity);
        detail::displayAddComponentEntry<PointSphereRenderComponent>("Point Sphere Renderer", entity);
        detail::displayAddComponentEntry<EdgeRenderComponent>("Edge Renderer", entity);
        detail::displayAddComponentEntry<EdgeCylinderRenderComponent>("Edge Cylinder Renderer", entity);
        detail::displayAddComponentEntry<CameraComponent>("Camera Component", entity);
        ImGui::EndPopup();
    }

    ImGui::PopItemWidth();

    detail::drawComponent<CameraComponent>(
        "Camera View",
        entity,
        [&](CameraComponent& camera_component)
        {
            atcg::ref_ptr<atcg::PerspectiveCamera> camera = camera_component.camera;
            if(entity.hasComponent<atcg::TransformComponent>())
            {
                auto& transform_component = entity.getComponent<atcg::TransformComponent>();
                camera->setView(glm::inverse(transform_component.getModel()));
            }

            float aspect_ratio = camera->getAspectRatio();

            label.str(std::string());
            label << "Aspect Ratio##" << id;
            ImGui::DragFloat(label.str().c_str(), &aspect_ratio, 0.05f, 0.1f, 5.0f);

            float fbo_aspect_ratio = (float)_camera_preview->width() / (float)_camera_preview->height();
            uint32_t height        = 128;
            uint32_t width         = (uint32_t)(aspect_ratio * 128.0f);

            if(fbo_aspect_ratio != aspect_ratio)
            {
                camera->setAspectRatio((float)width / (float)height);
                _camera_preview = atcg::make_ref<atcg::Framebuffer>(width, height);
                _camera_preview->attachColor();
                _camera_preview->attachDepth();
                _camera_preview->complete();
            }

            _camera_preview->use();
            atcg::Renderer::clear();
            atcg::Renderer::setViewport(0, 0, width, height);
            atcg::Renderer::draw(_scene, camera_component.camera);
            atcg::Renderer::getFramebuffer()->use();
            atcg::Renderer::setViewport(0,
                                        0,
                                        atcg::Renderer::getFramebuffer()->width(),
                                        atcg::Renderer::getFramebuffer()->height());

            uint64_t textureID = _camera_preview->getColorAttachement(0)->getID();

            ImVec2 window_size = ImGui::GetWindowSize();
            ImGui::SetCursorPos(ImVec2((window_size.x - width) * 0.5f, ImGui::GetCursorPosY()));
            ImGui::Image(reinterpret_cast<void*>(textureID), ImVec2(width, height), ImVec2 {0, 1}, ImVec2 {1, 0});

            if(ImGui::Button("Screenshot"))
            {
                // Create temporary framebuffer
                float width                                  = 1920.0f;
                float height                                 = width / aspect_ratio;
                atcg::ref_ptr<Framebuffer> screenshot_buffer = atcg::make_ref<Framebuffer>((int)width, (int)height);
                screenshot_buffer->attachColor();
                screenshot_buffer->attachDepth();
                screenshot_buffer->complete();

                screenshot_buffer->use();
                atcg::Renderer::clear();
                atcg::Renderer::setViewport(0, 0, width, height);
                atcg::Renderer::draw(_scene, camera_component.camera);
                atcg::Renderer::getFramebuffer()->use();
                atcg::Renderer::setViewport(0,
                                            0,
                                            atcg::Renderer::getFramebuffer()->width(),
                                            atcg::Renderer::getFramebuffer()->height());

                auto t  = std::time(nullptr);
                auto tm = *std::localtime(&t);
                std::ostringstream oss;
                oss << "bin/" << tag << "_" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S") << ".png";
                std::vector<uint8_t> buffer = atcg::Renderer::getFrame(screenshot_buffer);
                stbi_flip_vertically_on_write(true);
                stbi_write_png(oss.str().c_str(), (int)width, (int)height, 4, (void*)buffer.data(), 4 * (int)width);
            }
            atcg::Framebuffer::useDefault();

            if(ImGui::Button("Set from View"))
            {
                auto view = _scene->getAllEntitiesWith<EditorCameraComponent>();
                for(auto e: view)
                {
                    // Should only be one
                    Entity camera_entity(e, _scene.get());
                    atcg::ref_ptr<PerspectiveCamera> cam = camera_entity.getComponent<EditorCameraComponent>().camera;
                    camera->setView(cam->getView());
                    camera->setAspectRatio(cam->getAspectRatio());
                    if(entity.hasComponent<atcg::TransformComponent>())
                    {
                        auto& transform_component = entity.getComponent<atcg::TransformComponent>();
                        transform_component.setModel(glm::inverse(camera->getView()));
                    }
                }
            }
        });

    detail::drawComponent<TransformComponent>("Transform",
                                              entity,
                                              [&](TransformComponent& transform)
                                              {
                                                  glm::vec3 position = transform.getPosition();
                                                  label.str(std::string());
                                                  label << "Position##" << id;
                                                  if(ImGui::DragFloat3(label.str().c_str(), glm::value_ptr(position)))
                                                  {
                                                      transform.setPosition(position);
                                                  }
                                                  glm::vec3 scale = transform.getScale();
                                                  label.str(std::string());
                                                  label << "Scale##" << id;
                                                  if(ImGui::DragFloat3(label.str().c_str(), glm::value_ptr(scale)))
                                                  {
                                                      transform.setScale(scale);
                                                  }
                                                  glm::vec3 rotation = glm::degrees(transform.getRotation());
                                                  label.str(std::string());
                                                  label << "Rotation##" << id;
                                                  if(ImGui::DragFloat3(label.str().c_str(), glm::value_ptr(rotation)))
                                                  {
                                                      transform.setRotation(glm::radians(rotation));
                                                  }
                                              });
    detail::drawComponent<MeshRenderComponent>("Mesh Renderer",
                                               entity,
                                               [&](MeshRenderComponent& component)
                                               {
                                                   ImGui::Checkbox("Visible##visiblemesh", &component.visible);
                                                   glm::vec3 color = component.color;
                                                   label.str(std::string());
                                                   label << "Base Color##mesh" << id;
                                                   if(ImGui::ColorEdit3(label.str().c_str(), glm::value_ptr(color)))
                                                   {
                                                       component.color = color;
                                                   }
                                               });
    detail::drawComponent<PointRenderComponent>("Point Renderer",
                                                entity,
                                                [&](PointRenderComponent& component)
                                                {
                                                    ImGui::Checkbox("Visible##visiblepoints", &component.visible);
                                                    glm::vec3 color = component.color;
                                                    label.str(std::string());
                                                    label << "Base Color##point" << id;
                                                    if(ImGui::ColorEdit3(label.str().c_str(), glm::value_ptr(color)))
                                                    {
                                                        component.color = color;
                                                    }

                                                    float point_size = component.point_size;
                                                    label.str(std::string());
                                                    label << "Point Size##point" << id;
                                                    if(ImGui::DragFloat(label.str().c_str(), &point_size))
                                                    {
                                                        component.point_size = point_size;
                                                    }
                                                });
    detail::drawComponent<PointSphereRenderComponent>(
        "Point Sphere Renderer",
        entity,
        [&](PointSphereRenderComponent& component)
        {
            ImGui::Checkbox("Visible##visiblepointsphere", &component.visible);
            glm::vec3 color = component.color;
            label.str(std::string());
            label << "Base Color##pointsphere" << id;
            if(ImGui::ColorEdit3(label.str().c_str(), glm::value_ptr(color))) { component.color = color; }

            float point_size = component.point_size;
            label.str(std::string());
            label << "Point Size##pointsphere" << id;
            if(ImGui::DragFloat(label.str().c_str(), &point_size)) { component.point_size = point_size; }
        });

    detail::drawComponent<EdgeRenderComponent>("Edge Renderer",
                                               entity,
                                               [&](EdgeRenderComponent& component)
                                               {
                                                   ImGui::Checkbox("Visible##visibleedge", &component.visible);
                                                   glm::vec3 color = component.color;
                                                   label.str(std::string());
                                                   label << "Base Color##edge" << id;
                                                   if(ImGui::ColorEdit3(label.str().c_str(), glm::value_ptr(color)))
                                                   {
                                                       component.color = color;
                                                   }
                                               });

    detail::drawComponent<EdgeCylinderRenderComponent>(
        "Edge Cylinder Renderer",
        entity,
        [&](EdgeCylinderRenderComponent& component)
        {
            ImGui::Checkbox("Visible##visibleedgecylinder", &component.visible);
            glm::vec3 color = component.color;
            label.str(std::string());
            label << "Base Color##edgecylinder" << id;
            if(ImGui::ColorEdit3(label.str().c_str(), glm::value_ptr(color))) { component.color = color; }
            label.str(std::string());
            label << "Radius##edgecylinder" << id;
            float radius = component.radius;
            if(ImGui::DragFloat(label.str().c_str(), &radius)) { component.radius = radius; }
        });
}

SceneHierarchyPanel::SceneHierarchyPanel(const atcg::ref_ptr<Scene>& scene)
    : _scene(scene),
      _camera_preview(atcg::make_ref<atcg::Framebuffer>(128, 128))
{
    _camera_preview->attachColor();
    _camera_preview->attachDepth();
    _camera_preview->complete();
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

    if(ImGui::IsMouseDown(0) && ImGui::IsWindowHovered()) { _selected_entity = {}; }

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

    if(_selected_entity) { drawComponents(_selected_entity); }

    ImGui::End();
}
}    // namespace atcg