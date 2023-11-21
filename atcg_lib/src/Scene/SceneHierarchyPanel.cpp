#include <Scene/SceneHierarchyPanel.h>

#include <imgui.h>
#include <Scene/Components.h>
#include <portable-file-dialogs.h>

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

void displayMaterial(const std::string& key, Material& material)
{
    ImGui::Separator();

    ImGui::Text("Material");

    {
        auto spec        = material.getDiffuseTexture()->getSpecification();
        bool useTextures = spec.width != 1 || spec.height != 1;

        if(!useTextures)
        {
            auto diffuse = material.getDiffuseTexture()->getData();

            float color[4] = {(float)diffuse[0] / 255.0f,
                              (float)diffuse[1] / 255.0f,
                              (float)diffuse[2] / 255.0f,
                              (float)diffuse[3] / 255.0f};

            if(ImGui::ColorEdit4(("Diffuse##" + key).c_str(), color))
            {
                glm::vec4 new_color = glm::make_vec4(color);
                material.setDiffuseColor(new_color);
            }

            ImGui::SameLine();

            if(ImGui::Button(("...##diffuse" + key).c_str()))
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
                    auto img     = IO::imread(files[0], 2.2f);
                    auto texture = atcg::Texture2D::create(img);
                    material.setDiffuseTexture(texture);
                }
            }
        }
        else
        {
            ImGui::Text("Diffuse Texture");
            ImGui::SameLine();

            if(ImGui::Button(("X##diffuse" + key).c_str())) { material.setDiffuseColor(glm::vec4(1)); }
            else
                ImGui::Image((void*)(uint64_t)material.getDiffuseTexture()->getID(),
                             ImVec2(128, 128),
                             ImVec2 {0, 1},
                             ImVec2 {1, 0});
        }
    }

    {
        auto spec        = material.getNormalTexture()->getSpecification();
        bool useTextures = spec.width != 1 || spec.height != 1;

        if(!useTextures)
        {
            ImGui::Text("Normals");
            ImGui::SameLine();
            if(ImGui::Button(("...##normals" + key).c_str()))
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
                    auto img     = IO::imread(files[0]);
                    auto texture = atcg::Texture2D::create(img);
                    material.setNormalTexture(texture);
                }
            }
        }
        else
        {
            ImGui::Text("Normal Texture");
            ImGui::SameLine();

            if(ImGui::Button(("X##normal" + key).c_str())) { material.removeNormalMap(); }
            else
                ImGui::Image((void*)(uint64_t)material.getNormalTexture()->getID(),
                             ImVec2(128, 128),
                             ImVec2 {0, 1},
                             ImVec2 {1, 0});
        }
    }

    {
        auto spec        = material.getRoughnessTexture()->getSpecification();
        bool useTextures = spec.width != 1 || spec.height != 1;

        if(!useTextures)
        {
            auto data       = material.getRoughnessTexture()->getData();
            float roughness = *((float*)data.data());

            if(ImGui::DragFloat(("Roughness##" + key).c_str(), &roughness, 0.005f, 0.0f, 1.0f))
            {
                material.setRoughness(roughness);
            }

            ImGui::SameLine();

            if(ImGui::Button(("...##roughness" + key).c_str()))
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
                    auto img     = IO::imread(files[0]);
                    auto texture = atcg::Texture2D::create(img);
                    material.setRoughnessTexture(texture);
                }
            }
        }
        else
        {
            ImGui::Text("Roughness Texture");
            ImGui::SameLine();

            if(ImGui::Button(("X##roughness" + key).c_str())) { material.setRoughness(1.0f); }
            else
                ImGui::Image((void*)(uint64_t)material.getRoughnessTexture()->getID(),
                             ImVec2(128, 128),
                             ImVec2 {0, 1},
                             ImVec2 {1, 0});
        }
    }


    {
        auto spec        = material.getMetallicTexture()->getSpecification();
        bool useTextures = spec.width != 1 || spec.height != 1;

        if(!useTextures)
        {
            auto data      = material.getMetallicTexture()->getData();
            float metallic = *((float*)data.data());

            if(ImGui::DragFloat(("Metallic##" + key).c_str(), &metallic, 0.005f, 0.0f, 1.0f))
            {
                material.setMetallic(metallic);
            }

            ImGui::SameLine();

            if(ImGui::Button(("...##metallic" + key).c_str()))
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
                    auto img     = IO::imread(files[0]);
                    auto texture = atcg::Texture2D::create(img);
                    material.setMetallicTexture(texture);
                }
            }
        }
        else
        {
            ImGui::Text("Metallic Texture");
            ImGui::SameLine();

            if(ImGui::Button(("X##metallic" + key).c_str())) { material.setMetallic(0.0f); }
            else
                ImGui::Image((void*)(uint64_t)material.getMetallicTexture()->getID(),
                             ImVec2(128, 128),
                             ImVec2 {0, 1},
                             ImVec2 {1, 0});
        }
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

    detail::drawComponent<CameraComponent>(
        "Camera View",
        entity,
        [&](CameraComponent& camera_component)
        {
            atcg::ref_ptr<atcg::PerspectiveCamera> camera = camera_component.camera;
            bool has_transform                            = false;
            if(entity.hasComponent<atcg::TransformComponent>())
            {
                atcg::TransformComponent& transform_component = entity.getComponent<atcg::TransformComponent>();
                camera->setFromTransform(transform_component.getModel());
                has_transform = true;
            }

            float aspect_ratio = camera->getAspectRatio();
            float fov          = camera->getFOV();

            label.str(std::string());
            label << "Aspect Ratio##" << id;
            bool change_aspect = ImGui::DragFloat(label.str().c_str(), &aspect_ratio, 0.05f, 0.1f, 5.0f);
            bool change_fov    = ImGui::DragFloat(("FOV##" + id).c_str(), &fov, 0.5f, 10.0f, 120.0f);
            if(change_aspect || change_fov && has_transform)
            {
                atcg::TransformComponent& transform_component = entity.getComponent<atcg::TransformComponent>();
                glm::mat4 model                               = transform_component.getModel();
                float scale_x                                 = glm::length(model[0]);
                float scale_y                                 = glm::length(model[1]);
                float scale_z                                 = glm::length(model[2]);
                model                                         = model * glm::scale(glm::vec3(aspect_ratio / scale_x,
                                                     1.0f / scale_y,
                                                     glm::tan(glm::radians(fov) / 2.0f) / scale_z));
                transform_component.setModel(model);
                camera->setFOV(fov);
            }

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
                auto t  = std::time(nullptr);
                auto tm = *std::localtime(&t);
                std::ostringstream oss;
                oss << "bin/" << tag << "_" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S") << ".png";

                atcg::Renderer::screenshot(_scene, camera_component.camera, oss.str());
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

            label.str(std::string());
            label << "Color##" << id;
            ImGui::ColorEdit3(label.str().c_str(), glm::value_ptr(camera_component.color));
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

    detail::drawComponent<GeometryComponent>("Geometry",
                                             entity,
                                             [&](GeometryComponent& component)
                                             {
                                                 if(ImGui::Button("Import Mesh##GeometryComponent"))
                                                 {
                                                     auto f     = pfd::open_file("Choose files to read",
                                                                             pfd::path::home(),
                                                                                 {"Obj Files (.obj)", "*.obj"},
                                                                             pfd::opt::none);
                                                     auto files = f.result();
                                                     if(!files.empty())
                                                     {
                                                         auto mesh       = IO::read_mesh(files[0]);
                                                         component.graph = mesh;
                                                     }
                                                 }

                                                 // glm::vec3 target   = glm::vec3(0);
                                                 // glm::vec3 location = glm::vec3(2, 2, -2);
                                                 // glm::mat4 model    = glm::mat4(1);
                                                 // bool reset_model   = false;
                                                 // if(entity.hasComponent<TransformComponent>())
                                                 // {
                                                 //     auto& transform = entity.getComponent<TransformComponent>();
                                                 //     model           = transform.getModel();
                                                 //     transform.setModel(glm::mat4(1));
                                                 //     reset_model = true;
                                                 // }

                                                 // _object_preview_cam->setPosition(location);
                                                 // _object_preview_cam->setLookAt(target);

                                                 // _object_preview->use();
                                                 // atcg::Renderer::clear();
                                                 // atcg::Renderer::setViewport(0, 0, 128, 128);
                                                 // atcg::Renderer::draw(entity, _object_preview_cam);
                                                 // atcg::Renderer::getFramebuffer()->use();
                                                 // atcg::Renderer::setViewport(0,
                                                 //                             0,
                                                 //                             atcg::Renderer::getFramebuffer()->width(),
                                                 //                             atcg::Renderer::getFramebuffer()->height());

                                                 // uint64_t textureID =
                                                 // _object_preview->getColorAttachement(0)->getID();

                                                 // ImVec2 window_size = ImGui::GetWindowSize();
                                                 // ImGui::SetCursorPos(ImVec2((window_size.x - 128) * 0.5f,
                                                 // ImGui::GetCursorPosY()));
                                                 // ImGui::Image(reinterpret_cast<void*>(textureID), ImVec2(128, 128),
                                                 // ImVec2 {0, 1}, ImVec2 {1, 0});

                                                 // atcg::Framebuffer::useDefault();

                                                 // if(reset_model)
                                                 // {
                                                 //     auto& transform = entity.getComponent<TransformComponent>();
                                                 //     transform.setModel(model);
                                                 // }
                                             });

    detail::drawComponent<MeshRenderComponent>("Mesh Renderer",
                                               entity,
                                               [&](MeshRenderComponent& component)
                                               {
                                                   ImGui::Checkbox("Visible##visiblemesh", &component.visible);

                                                   // Material
                                                   Material& material = component.material;

                                                   detail::displayMaterial("mesh", material);
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
    detail::drawComponent<PointSphereRenderComponent>("Point Sphere Renderer",
                                                      entity,
                                                      [&](PointSphereRenderComponent& component)
                                                      {
                                                          ImGui::Checkbox("Visible##visiblepointsphere",
                                                                          &component.visible);

                                                          float point_size = component.point_size;
                                                          label.str(std::string());
                                                          label << "Point Size##pointsphere" << id;
                                                          if(ImGui::DragFloat(label.str().c_str(), &point_size))
                                                          {
                                                              component.point_size = point_size;
                                                          }

                                                          // Material
                                                          Material& material = component.material;

                                                          detail::displayMaterial("pointsphere", material);
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

    detail::drawComponent<EdgeCylinderRenderComponent>("Edge Cylinder Renderer",
                                                       entity,
                                                       [&](EdgeCylinderRenderComponent& component)
                                                       {
                                                           ImGui::Checkbox("Visible##visibleedgecylinder",
                                                                           &component.visible);
                                                           label.str(std::string());
                                                           label << "Radius##edgecylinder" << id;
                                                           float radius = component.radius;
                                                           if(ImGui::DragFloat(label.str().c_str(), &radius))
                                                           {
                                                               component.radius = radius;
                                                           }

                                                           // Material
                                                           Material& material = component.material;

                                                           detail::displayMaterial("edgecylinder", material);
                                                       });

    detail::drawComponent<InstanceRenderComponent>("Instance Renderer",
                                                   entity,
                                                   [&](InstanceRenderComponent& component)
                                                   {
                                                       ImGui::Checkbox("Visible##visibleinstance", &component.visible);

                                                       // Material
                                                       Material& material = component.material;

                                                       detail::displayMaterial("instance", material);
                                                   });
}

void SceneHierarchyPanel::drawSceneProperties()
{
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
                         ImVec2(128, 64),
                         ImVec2 {0, 1},
                         ImVec2 {1, 0});
            if(ImGui::Button("Remove skybox##skybox")) { Renderer::removeSkybox(); }
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
      _camera_preview(atcg::make_ref<atcg::Framebuffer>(128, 128)),
      _object_preview(atcg::make_ref<atcg::Framebuffer>(128, 128)),
      _object_preview_cam(atcg::make_ref<atcg::PerspectiveCamera>(1.0f))
{
    _camera_preview->attachColor();
    _camera_preview->attachDepth();
    _camera_preview->complete();

    _object_preview->attachColor();
    _object_preview->attachDepth();
    _object_preview->complete();
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

    if(ImGui::BeginTabBar("TabBarComponents"))
    {
        if(ImGui::BeginTabItem("Components"))
        {
            if(_selected_entity) { drawComponents(_selected_entity); }
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