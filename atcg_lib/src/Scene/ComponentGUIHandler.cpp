#include <Scene/ComponentGUIHandler.h>

#include <Core/Application.h>
#include <Scene/Entity.h>

#include <imgui.h>
#include <portable-file-dialogs.h>

namespace atcg
{
template<typename T>
void ComponentGUIHandler::draw_component(Entity entity, T& component)
{
}

template<>
void ComponentGUIHandler::draw_component<TransformComponent>(Entity entity, TransformComponent& transform)
{
    std::string id = std::to_string(entity.getComponent<IDComponent>().ID);

    glm::vec3 position = transform.getPosition();
    std::stringstream label;
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
}

template<>
void ComponentGUIHandler::draw_component<CameraComponent>(Entity entity, CameraComponent& camera_component)
{
    float content_scale = atcg::Application::get()->getWindow()->getContentScale();
    std::string id      = std::to_string(entity.getComponent<IDComponent>().ID);

    atcg::ref_ptr<atcg::PerspectiveCamera> camera =
        std::dynamic_pointer_cast<atcg::PerspectiveCamera>(camera_component.camera);

    float aspect_ratio = camera->getAspectRatio();
    float fov          = camera->getFOV();

    std::stringstream label;
    label << "Aspect Ratio##" << id;
    if(ImGui::DragFloat(label.str().c_str(), &aspect_ratio, 0.05f, 0.1f, 5.0f))
    {
        camera->setAspectRatio(aspect_ratio);
    }

    if(ImGui::DragFloat(("FOV##" + id).c_str(), &fov, 0.5f, 10.0f, 120.0f))
    {
        camera->setFOV(fov);
    }

    float fbo_aspect_ratio = (float)_camera_preview->width() / (float)_camera_preview->height();
    uint32_t height        = 128;
    uint32_t width         = (uint32_t)(aspect_ratio * 128.0f);

    if(glm::abs(fbo_aspect_ratio - aspect_ratio) > 1e-5f)
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
    atcg::Renderer::setDefaultViewport();

    uint64_t textureID = _camera_preview->getColorAttachement(0)->getID();

    ImVec2 window_size = ImGui::GetWindowSize();
    ImGui::SetCursorPos(ImVec2((window_size.x - width) * 0.5f, ImGui::GetCursorPosY()));
    ImGui::Image(reinterpret_cast<void*>(textureID),
                 ImVec2(content_scale * width, content_scale * height),
                 ImVec2 {0, 1},
                 ImVec2 {1, 0});

    if(ImGui::Button("Screenshot"))
    {
        auto t  = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;

        std::string tag = entity.getComponent<NameComponent>().name;

        oss << "bin/" << tag << "_" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S") << ".png";

        atcg::Renderer::screenshot(_scene, camera_component.camera, 1920, oss.str());
    }
    atcg::Framebuffer::useDefault();

    if(ImGui::Button("Set from View"))
    {
        auto view = _scene->getAllEntitiesWith<EditorCameraComponent>();
        for(auto e: view)
        {
            // Should only be one
            Entity camera_entity(e, _scene.get());
            atcg::ref_ptr<PerspectiveCamera> cam = std::dynamic_pointer_cast<PerspectiveCamera>(
                camera_entity.getComponent<EditorCameraComponent>().camera);
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

    // Display camera extrinsic/intrinsic as transform
    atcg::TransformComponent transform(camera->getAsTransform());
    draw_component(entity, transform);

    // May be updated
    // ? Optimize this
    camera->setFromTransform(transform);
}

template<>
void ComponentGUIHandler::draw_component<GeometryComponent>(Entity entity, GeometryComponent& component)
{
    if(ImGui::Button("Import Mesh##GeometryComponent"))
    {
        auto f =
            pfd::open_file("Choose files to read", pfd::path::home(), {"Obj Files (.obj)", "*.obj"}, pfd::opt::none);
        auto files = f.result();
        if(!files.empty())
        {
            auto mesh       = IO::read_mesh(files[0]);
            component.graph = mesh;
        }
    }
}

template<>
void ComponentGUIHandler::draw_component<MeshRenderComponent>(Entity entity, MeshRenderComponent& component)
{
    ImGui::Checkbox("Visible##visiblemesh", &component.visible);

    // Material
    Material& material = component.material;

    displayMaterial("mesh", material);
}

template<>
void ComponentGUIHandler::draw_component<PointRenderComponent>(Entity entity, PointRenderComponent& component)
{
    std::string id = std::to_string(entity.getComponent<IDComponent>().ID);

    ImGui::Checkbox("Visible##visiblepoints", &component.visible);
    glm::vec3 color = component.color;
    std::stringstream label;
    label << "Base Color##point" << id;
    if(ImGui::ColorEdit3(label.str().c_str(), glm::value_ptr(color)))
    {
        component.color = color;
    }

    int point_size = (int)component.point_size;
    label.str(std::string());
    label << "Point Size##point" << id;
    if(ImGui::DragInt(label.str().c_str(), &point_size, 1, 1, INT_MAX))
    {
        component.point_size = (float)point_size;
    }
}

template<>
void ComponentGUIHandler::draw_component<PointSphereRenderComponent>(Entity entity,
                                                                     PointSphereRenderComponent& component)
{
    std::string id = std::to_string(entity.getComponent<IDComponent>().ID);

    ImGui::Checkbox("Visible##visiblepointsphere", &component.visible);

    float point_size = component.point_size;
    std::stringstream label;
    label << "Point Size##pointsphere" << id;
    if(ImGui::DragFloat(label.str().c_str(), &point_size, 0.001f, 0.001f, FLT_MAX / INT_MAX))
    {
        component.point_size = point_size;
    }

    // Material
    Material& material = component.material;

    displayMaterial("pointsphere", material);
}

template<>
void ComponentGUIHandler::draw_component<EdgeRenderComponent>(Entity entity, EdgeRenderComponent& component)
{
    std::string id = std::to_string(entity.getComponent<IDComponent>().ID);

    ImGui::Checkbox("Visible##visibleedge", &component.visible);
    glm::vec3 color = component.color;
    std::stringstream label;
    label << "Base Color##edge" << id;
    if(ImGui::ColorEdit3(label.str().c_str(), glm::value_ptr(color)))
    {
        component.color = color;
    }
}

template<>
void ComponentGUIHandler::draw_component<EdgeCylinderRenderComponent>(Entity entity,
                                                                      EdgeCylinderRenderComponent& component)
{
    std::string id = std::to_string(entity.getComponent<IDComponent>().ID);

    ImGui::Checkbox("Visible##visibleedgecylinder", &component.visible);
    std::stringstream label;
    label << "Radius##edgecylinder" << id;
    float radius = component.radius;
    if(ImGui::DragFloat(label.str().c_str(), &radius, 0.001f, 0.001f, FLT_MAX / INT_MAX))
    {
        component.radius = radius;
    }

    // Material
    Material& material = component.material;

    displayMaterial("edgecylinder", material);
}

template<>
void ComponentGUIHandler::draw_component<InstanceRenderComponent>(Entity entity, InstanceRenderComponent& component)
{
    ImGui::Checkbox("Visible##visibleinstance", &component.visible);

    // Material
    Material& material = component.material;

    displayMaterial("instance", material);
}

void ComponentGUIHandler::displayMaterial(const std::string& key, Material& material)
{
    float content_scale = atcg::Application::get()->getWindow()->getContentScale();
    ImGui::Separator();

    ImGui::Text("Material");

    {
        auto spec        = material.getDiffuseTexture()->getSpecification();
        bool useTextures = spec.width != 1 || spec.height != 1;

        if(!useTextures)
        {
            auto diffuse = material.getDiffuseTexture()->getData(atcg::CPU);

            float color[4] = {diffuse.index({0, 0, 0}).item<float>() / 255.0f,
                              diffuse.index({0, 0, 1}).item<float>() / 255.0f,
                              diffuse.index({0, 0, 2}).item<float>() / 255.0f,
                              diffuse.index({0, 0, 3}).item<float>() / 255.0f};

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

            if(ImGui::Button(("X##diffuse" + key).c_str()))
            {
                material.setDiffuseColor(glm::vec4(1));
            }
            else
                ImGui::Image((void*)(uint64_t)material.getDiffuseTexture()->getID(),
                             ImVec2(content_scale * 128, content_scale * 128),
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

            if(ImGui::Button(("X##normal" + key).c_str()))
            {
                material.removeNormalMap();
            }
            else
                ImGui::Image((void*)(uint64_t)material.getNormalTexture()->getID(),
                             ImVec2(content_scale * 128, content_scale * 128),
                             ImVec2 {0, 1},
                             ImVec2 {1, 0});
        }
    }

    {
        auto spec        = material.getRoughnessTexture()->getSpecification();
        bool useTextures = spec.width != 1 || spec.height != 1;

        if(!useTextures)
        {
            auto data       = material.getRoughnessTexture()->getData(atcg::CPU);
            float roughness = data.item<float>();

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

            if(ImGui::Button(("X##roughness" + key).c_str()))
            {
                material.setRoughness(1.0f);
            }
            else
                ImGui::Image((void*)(uint64_t)material.getRoughnessTexture()->getID(),
                             ImVec2(content_scale * 128, content_scale * 128),
                             ImVec2 {0, 1},
                             ImVec2 {1, 0});
        }
    }


    {
        auto spec        = material.getMetallicTexture()->getSpecification();
        bool useTextures = spec.width != 1 || spec.height != 1;

        if(!useTextures)
        {
            auto data      = material.getMetallicTexture()->getData(atcg::CPU);
            float metallic = data.item<float>();

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

            if(ImGui::Button(("X##metallic" + key).c_str()))
            {
                material.setMetallic(0.0f);
            }
            else
                ImGui::Image((void*)(uint64_t)material.getMetallicTexture()->getID(),
                             ImVec2(content_scale * 128, content_scale * 128),
                             ImVec2 {0, 1},
                             ImVec2 {1, 0});
        }
    }
}
}    // namespace atcg