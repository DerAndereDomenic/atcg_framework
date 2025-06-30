#include <Scene/ComponentGUIHandler.h>

#include <Core/Application.h>
#include <Scene/Entity.h>
#include <Math/Utils.h>

#include <imgui.h>
#include <portable-file-dialogs.h>

namespace atcg
{
namespace GUI
{

void ComponentGUIRenderer<TransformComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                              Entity entity,
                                                              TransformComponent& transform) const
{
    std::string id = std::to_string(entity.getComponent<IDComponent>().ID());

    glm::vec3 position = transform.getPosition();
    std::stringstream label;
    label << "Position##" << id;
    if(ImGui::DragFloat3(label.str().c_str(), glm::value_ptr(position), 0.05f))
    {
        RevisionStack::startRecording<ComponentEditedRevision<TransformComponent>>(scene, entity);
        transform.setPosition(position);
        atcg::RevisionStack::endRecording();
    }
    glm::vec3 scale = transform.getScale();
    label.str(std::string());
    label << "Scale##" << id;
    if(ImGui::DragFloat3(label.str().c_str(), glm::value_ptr(scale), 0.05f, 1e-5f, FLT_MAX))
    {
        RevisionStack::startRecording<ComponentEditedRevision<TransformComponent>>(scene, entity);
        scale = glm::clamp(scale, 1e-5f, FLT_MAX);
        transform.setScale(scale);
        atcg::RevisionStack::endRecording();
    }
    glm::vec3 rotation = glm::degrees(transform.getRotation());
    label.str(std::string());
    label << "Rotation##" << id;
    if(ImGui::DragFloat3(label.str().c_str(), glm::value_ptr(rotation), 0.05f))
    {
        RevisionStack::startRecording<ComponentEditedRevision<TransformComponent>>(scene, entity);
        transform.setRotation(glm::radians(rotation));
        atcg::RevisionStack::endRecording();
    }

    if(entity.hasComponent<atcg::GeometryComponent>())
    {
        if(ImGui::Button("Apply Transform"))
        {
            RevisionStack::startRecording<
                UnionRevision<ComponentEditedRevision<TransformComponent>, ComponentEditedRevision<GeometryComponent>>>(
                scene,
                entity);
            auto& geometry = entity.getComponent<atcg::GeometryComponent>();
            auto graph     = geometry.graph()->copy();
            applyTransform(graph, transform);
            geometry.setGraph(graph);
            atcg::RevisionStack::endRecording();
        }

        if(ImGui::Button("Normalize"))
        {
            RevisionStack::startRecording<
                UnionRevision<ComponentEditedRevision<TransformComponent>, ComponentEditedRevision<GeometryComponent>>>(
                scene,
                entity);
            auto& geometry = entity.getComponent<atcg::GeometryComponent>();
            auto graph     = geometry.graph()->copy();
            normalize(graph, transform);
            geometry.setGraph(graph);
            atcg::RevisionStack::endRecording();
        }
    }
}


void ComponentGUIRenderer<CameraComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                           Entity entity,
                                                           CameraComponent& _component) const
{
    CameraComponent component = _component;
    bool updated              = false;

    float content_scale = atcg::Application::get()->getWindow()->getContentScale();
    std::string id      = std::to_string(entity.getComponent<IDComponent>().ID());

    atcg::ref_ptr<atcg::PerspectiveCamera> camera =
        std::dynamic_pointer_cast<atcg::PerspectiveCamera>(component.camera->copy());

    atcg::CameraIntrinsics intrinsics = camera->getIntrinsics();
    glm::mat3 K = atcg::CameraUtils::convert_to_opencv(intrinsics, component.width, component.height);

    float fx = K[0][0];
    float fy = K[1][1];
    float cx = K[0][2];
    float cy = K[1][2];

    float f[2]      = {fx, fy};
    float c[2]      = {cx, cy};
    uint32_t res[2] = {component.width, component.height};
    float offset[2] = {intrinsics.opticalCenter().x, intrinsics.opticalCenter().y};

    float aspect_ratio = intrinsics.aspectRatio();
    float fov          = intrinsics.FOV();

    std::stringstream label;
    label << "Aspect Ratio##" << id;
    if(ImGui::DragFloat(label.str().c_str(), &aspect_ratio, 0.05f, 0.1f, 5.0f))
    {
        intrinsics.setAspectRatio(aspect_ratio);
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
    label << "Resolution##" << id;
    if(ImGui::DragInt2(label.str().c_str(), (int*)res, 1, 1, 4096))
    {
        component.width  = res[0];
        component.height = res[1];
        updated          = true;
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
                                                            component.width,
                                                            component.height);
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
                                                            component.width,
                                                            component.height);
        updated    = true;
    }

    label.str(std::string());
    label << "Color##" << id;
    updated = ImGui::ColorEdit3(label.str().c_str(), glm::value_ptr(component.color)) || updated;

    label.str(std::string());
    label << "Scale##" << id;
    updated = ImGui::DragFloat(label.str().c_str(), &component.render_scale, 0.01f, 0.01f, FLT_MAX) || updated;

    uint32_t preview_height = 128;
    uint32_t preview_width =
        glm::clamp((uint32_t)(float(component.width) / float(component.height) * 128.0f), uint32_t(1), uint32_t(4096));

    if(!component.preview || component.preview->width() != preview_width ||
       component.preview->height() != preview_height)
    {
        component.preview = atcg::make_ref<atcg::Framebuffer>(preview_width, preview_height);
        component.preview->attachColor();
        component.preview->attachDepth();
        component.preview->complete();
        updated = true;
    }

    component.preview->use();
    atcg::Renderer::clear();
    atcg::Renderer::setViewport(0, 0, preview_width, preview_height);

    atcg::Dictionary context;
    context.setValue("camera", component.camera);
    scene->draw(context);
    atcg::Renderer::useScreenBuffer();
    atcg::Renderer::setDefaultViewport();

    updated = ImGui::Checkbox("Show Preview##cam", &component.render_preview) || updated;

    uint64_t textureID = component.preview->getColorAttachement(0)->getID();

    ImVec2 window_size = ImGui::GetWindowSize();
    ImGui::SetCursorPos(ImVec2((window_size.x - preview_width) * 0.5f, ImGui::GetCursorPosY()));
    ImGui::Image((ImTextureID)textureID,
                 ImVec2(content_scale * preview_width, content_scale * preview_height),
                 ImVec2 {0, 1},
                 ImVec2 {1, 0});

    if(ImGui::Button("Screenshot"))
    {
        auto t  = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;

        std::string_view tag = entity.getComponent<NameComponent>().name();

        oss << "bin/" << tag << "_" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S") << ".png";

        atcg::Renderer::screenshot(scene, component.camera, component.width, component.height, oss.str());
    }
    atcg::Framebuffer::useDefault();

    ImGui::Separator();

    if(!component.image)
    {
        ImGui::Text("Image");
        ImGui::SameLine();
        if(ImGui::Button("...##imageload"))
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
                auto img        = IO::imread(files[0]);
                component.image = atcg::Texture2D::create(img);
                updated         = true;
            }
        }
    }
    else
    {
        ImGui::Text("Image");
        ImGui::SameLine();

        if(ImGui::Button("X##imagedelete"))
        {
            component.image = nullptr;
            updated         = true;
        }
        else
        {
            float aspect             = float(component.image->width()) / float(component.image->height());
            uint32_t preview_height_ = 128;
            uint32_t preview_width_  = glm::clamp((uint32_t)(aspect * 128.0f), uint32_t(1), uint32_t(4096));
            ImGui::SetCursorPos(ImVec2((window_size.x - preview_width_) * 0.5f, ImGui::GetCursorPosY()));
            ImGui::Image((ImTextureID)component.image->getID(),
                         ImVec2(content_scale * preview_width_, content_scale * preview_height_),
                         ImVec2 {0, 1},
                         ImVec2 {1, 0});
        }
    }

    ImGui::Separator();

    auto scene_camera = scene->getCamera();

    if(scene_camera)
    {
        if(ImGui::Button("Fly to"))
        {
            scene_camera->setExtrinsics(component.camera->getExtrinsics());
        }

        if(ImGui::Button("Set from View"))
        {
            camera->setExtrinsics(scene_camera->getExtrinsics());
            intrinsics = scene_camera->getIntrinsics();

            component.width  = atcg::Renderer::getFramebuffer()->width();
            component.height = atcg::Renderer::getFramebuffer()->height();

            if(entity.hasComponent<atcg::TransformComponent>())
            {
                entity.getComponent<atcg::TransformComponent>().setModel(glm::inverse(scene_camera->getView()));
            }

            updated = true;
        }
    }

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<CameraComponent>>(scene, entity);
        _component = component;
        camera->setIntrinsics(intrinsics);
        _component.camera = camera;
        atcg::RevisionStack::endRecording();
    }

    if(entity.hasComponent<atcg::TransformComponent>())
    {
        glm::mat4 model = entity.getComponent<atcg::TransformComponent>().getModel();

        float scale_x = glm::length(glm::vec3(model[0]));
        float scale_y = glm::length(glm::vec3(model[1]));
        float scale_z = glm::length(glm::vec3(model[2]));

        model = model * glm::scale(glm::vec3(1.0f / scale_x, 1.0f / scale_y, 1.0f / scale_z));

        camera->setView(glm::inverse(model));
        _component.camera = camera;
    }
}


void ComponentGUIRenderer<GeometryComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                             Entity entity,
                                                             GeometryComponent& component) const
{
    if(ImGui::Button("Import Mesh##GeometryComponent"))
    {
        auto f =
            pfd::open_file("Choose files to read", pfd::path::home(), {"Obj Files (.obj)", "*.obj"}, pfd::opt::none);
        auto files = f.result();
        if(!files.empty())
        {
            RevisionStack::startRecording<ComponentEditedRevision<GeometryComponent>>(scene, entity);
            auto mesh = IO::read_any(files[0]);
            component.setGraph(mesh);
            atcg::RevisionStack::endRecording();
        }
    }
}


void ComponentGUIRenderer<MeshRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               MeshRenderComponent& component) const
{
    MeshRenderComponent component_copy = component;
    bool updated                       = ImGui::Checkbox("Visible##visiblemesh", &component_copy.visible);

    // Material
    auto material = component_copy.material();

    updated = displayMaterial("mesh", material) || updated;
    updated = ImGui::Checkbox("Receive Shadows##MeshRenderComponent", &component_copy.receive_shadow) || updated;

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<MeshRenderComponent>>(scene, entity);
        component = component_copy;
        atcg::RevisionStack::endRecording();
    }
}


void ComponentGUIRenderer<PointRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                                Entity entity,
                                                                PointRenderComponent& _component) const
{
    std::string id = std::to_string(entity.getComponent<IDComponent>().ID());

    PointRenderComponent component = _component;

    bool updated    = ImGui::Checkbox("Visible##visiblepoints", &component.visible);
    glm::vec3 color = component.color;
    std::stringstream label;
    label << "Base Color##point" << id;
    if(ImGui::ColorEdit3(label.str().c_str(), glm::value_ptr(color)))
    {
        component.color = color;
        updated         = true;
    }

    int point_size = (int)component.point_size;
    label.str(std::string());
    label << "Point Size##point" << id;
    if(ImGui::DragInt(label.str().c_str(), &point_size, 1, 1, INT_MAX))
    {
        component.point_size = (float)point_size;
        updated              = true;
    }

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<PointRenderComponent>>(scene, entity);
        _component = component;
        atcg::RevisionStack::endRecording();
    }
}


void ComponentGUIRenderer<PointSphereRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                                      Entity entity,
                                                                      PointSphereRenderComponent& _component) const
{
    PointSphereRenderComponent component = _component;
    std::string id                       = std::to_string(entity.getComponent<IDComponent>().ID());

    bool updated = ImGui::Checkbox("Visible##visiblepointsphere", &component.visible);

    float point_size = component.point_size;
    std::stringstream label;
    label << "Point Size##pointsphere" << id;
    if(ImGui::DragFloat(label.str().c_str(), &point_size, 0.001f, 0.001f, FLT_MAX / INT_MAX))
    {
        component.point_size = point_size;
        updated              = true;
    }

    // Material
    auto material = component.material();

    updated = displayMaterial("pointsphere", material) || updated;

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<PointSphereRenderComponent>>(scene, entity);
        _component = component;
        atcg::RevisionStack::endRecording();
    }
}


void ComponentGUIRenderer<EdgeRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               EdgeRenderComponent& _component) const
{
    EdgeRenderComponent component = _component;

    std::string id = std::to_string(entity.getComponent<IDComponent>().ID());

    bool updated    = ImGui::Checkbox("Visible##visibleedge", &component.visible);
    glm::vec3 color = component.color;
    std::stringstream label;
    label << "Base Color##edge" << id;
    if(ImGui::ColorEdit3(label.str().c_str(), glm::value_ptr(color)))
    {
        component.color = color;
        updated         = true;
    }

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<EdgeRenderComponent>>(scene, entity);
        _component = component;
        atcg::RevisionStack::endRecording();
    }
}


void ComponentGUIRenderer<EdgeCylinderRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                                       Entity entity,
                                                                       EdgeCylinderRenderComponent& _component) const
{
    EdgeCylinderRenderComponent component = _component;
    std::string id                        = std::to_string(entity.getComponent<IDComponent>().ID());

    bool updated = ImGui::Checkbox("Visible##visibleedgecylinder", &component.visible);
    std::stringstream label;
    label << "Radius##edgecylinder" << id;
    float radius = component.radius;
    if(ImGui::DragFloat(label.str().c_str(), &radius, 0.001f, 0.001f, FLT_MAX / INT_MAX))
    {
        component.radius = radius;
        updated          = true;
    }

    // Material
    auto material = component.material();

    updated = displayMaterial("edgecylinder", material) || updated;

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<EdgeCylinderRenderComponent>>(scene, entity);
        _component = component;
        atcg::RevisionStack::endRecording();
    }
}


void ComponentGUIRenderer<InstanceRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                                   Entity entity,
                                                                   InstanceRenderComponent& _component) const
{
    InstanceRenderComponent component = _component;
    bool updated                      = ImGui::Checkbox("Visible##visibleinstance", &component.visible);

    // Material
    auto material = component.material();

    updated = displayMaterial("instance", material) || updated;
    updated = ImGui::Checkbox("Receive Shadows##InstanceRenderComponent", &component.receive_shadow) || updated;

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<InstanceRenderComponent>>(scene, entity);
        _component = component;
        atcg::RevisionStack::endRecording();
    }
}


void ComponentGUIRenderer<PointLightComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               PointLightComponent& _component) const
{
    PointLightComponent component = _component;
    bool updated = ImGui::DragFloat("Intensity##PointLight", &component.intensity, 0.01f, 0.0f, FLT_MAX);
    updated      = ImGui::ColorEdit3("Color##PointLight", glm::value_ptr(component.color)) || updated;
    updated      = ImGui::Checkbox("Cast Shadows##PointLight", &component.cast_shadow) || updated;

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<PointLightComponent>>(scene, entity);
        _component = component;
        atcg::RevisionStack::endRecording();
    }
}


void ComponentGUIRenderer<ScriptComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                           Entity entity,
                                                           ScriptComponent& _component) const
{
    ScriptComponent component = _component;
    bool updated              = false;

    if(component.script == nullptr)
    {
        if(ImGui::Button("Load Script"))
        {
            updated = true;

            auto f     = pfd::open_file("Choose files to read",
                                    pfd::path::home(),
                                        {"Python Files (.py)", "*.py"},
                                    pfd::opt::none);
            auto files = f.result();
            if(!files.empty())
            {
                component.script = atcg::make_ref<atcg::PythonScript>(files[0]);
                component.script->init(scene, entity);
                component.script->onAttach();
            }
        }
    }
    else
    {
        if(ImGui::Button("X"))
        {
            component.script->onDetach();
            component.script = nullptr;
            updated          = true;
        }
        else
        {
            ImGui::SameLine();
            ImGui::Text(component.script->getFilePath().string().c_str());
        }
    }

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<ScriptComponent>>(scene, entity);
        _component = component;
        atcg::RevisionStack::endRecording();
    }
}

bool displayMaterial(const std::string& key, const atcg::ref_ptr<Material>& material)
{
    bool updated = false;

    float content_scale = atcg::Application::get()->getWindow()->getContentScale();
    ImGui::Separator();

    ImGui::Text("Material");

    {
        auto spec        = material->getDiffuseTexture()->getSpecification();
        bool useTextures = spec.width != 1 || spec.height != 1;

        if(!useTextures)
        {
            auto diffuse = material->getDiffuseTexture()->getData(atcg::CPU);

            float color[4] = {diffuse.index({0, 0, 0}).item<float>() / 255.0f,
                              diffuse.index({0, 0, 1}).item<float>() / 255.0f,
                              diffuse.index({0, 0, 2}).item<float>() / 255.0f,
                              diffuse.index({0, 0, 3}).item<float>() / 255.0f};

            if(ImGui::ColorEdit4(("Diffuse##" + key).c_str(), color))
            {
                glm::vec4 new_color = glm::make_vec4(color);
                material->setDiffuseColor(new_color);
                updated = true;
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
                    material->setDiffuseTexture(texture);
                    updated = true;
                }
            }
        }
        else
        {
            ImGui::Text("Diffuse Texture");
            ImGui::SameLine();

            if(ImGui::Button(("X##diffuse" + key).c_str()))
            {
                material->setDiffuseColor(glm::vec4(1));
                updated = true;
            }
            else
                ImGui::Image((ImTextureID)material->getDiffuseTexture()->getID(),
                             ImVec2(content_scale * 128, content_scale * 128),
                             ImVec2 {0, 1},
                             ImVec2 {1, 0});
        }
    }

    {
        auto spec        = material->getNormalTexture()->getSpecification();
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
                    material->setNormalTexture(texture);
                    updated = true;
                }
            }
        }
        else
        {
            ImGui::Text("Normal Texture");
            ImGui::SameLine();

            if(ImGui::Button(("X##normal" + key).c_str()))
            {
                material->removeNormalMap();
                updated = true;
            }
            else
                ImGui::Image((ImTextureID)material->getNormalTexture()->getID(),
                             ImVec2(content_scale * 128, content_scale * 128),
                             ImVec2 {0, 1},
                             ImVec2 {1, 0});
        }
    }

    {
        auto spec        = material->getRoughnessTexture()->getSpecification();
        bool useTextures = spec.width != 1 || spec.height != 1;

        if(!useTextures)
        {
            auto data       = material->getRoughnessTexture()->getData(atcg::CPU);
            float roughness = data.item<float>();

            if(ImGui::DragFloat(("Roughness##" + key).c_str(), &roughness, 0.005f, 0.0f, 1.0f))
            {
                material->setRoughness(roughness);
                updated = true;
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
                    material->setRoughnessTexture(texture);
                    updated = true;
                }
            }
        }
        else
        {
            ImGui::Text("Roughness Texture");
            ImGui::SameLine();

            if(ImGui::Button(("X##roughness" + key).c_str()))
            {
                material->setRoughness(1.0f);
                updated = true;
            }
            else
                ImGui::Image((ImTextureID)material->getRoughnessTexture()->getID(),
                             ImVec2(content_scale * 128, content_scale * 128),
                             ImVec2 {0, 1},
                             ImVec2 {1, 0});
        }
    }


    {
        auto spec        = material->getMetallicTexture()->getSpecification();
        bool useTextures = spec.width != 1 || spec.height != 1;

        if(!useTextures)
        {
            auto data      = material->getMetallicTexture()->getData(atcg::CPU);
            float metallic = data.item<float>();

            if(ImGui::DragFloat(("Metallic##" + key).c_str(), &metallic, 0.005f, 0.0f, 1.0f))
            {
                material->setMetallic(metallic);
                updated = true;
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
                    material->setMetallicTexture(texture);
                    updated = true;
                }
            }
        }
        else
        {
            ImGui::Text("Metallic Texture");
            ImGui::SameLine();

            if(ImGui::Button(("X##metallic" + key).c_str()))
            {
                material->setMetallic(0.0f);
                updated = true;
            }
            else
                ImGui::Image((ImTextureID)material->getMetallicTexture()->getID(),
                             ImVec2(content_scale * 128, content_scale * 128),
                             ImVec2 {0, 1},
                             ImVec2 {1, 0});
        }
    }

    return updated;
}
}    // namespace GUI
}    // namespace atcg