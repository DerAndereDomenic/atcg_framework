#include <Scene/Components/CameraComponent.h>
#include <Scene/ComponentRegistry.h>

#include <Core/Application.h>
#include <Utils/Utils.h>

namespace atcg
{

void ComponentRenderer<CameraComponent>::renderComponent(atcg::RendererSystem* _renderer,
                                                         Entity entity,
                                                         const atcg::ref_ptr<Camera>& camera,
                                                         atcg::Dictionary& auxiliary) const
{
    bool draw_cameras = auxiliary.getValueOr<bool>("draw_cameras", true);
    if(!draw_cameras)
    {
        return;
    }

    auto camera_frustum       = AssetManager::getCameraFrustumMesh();
    auto quad                 = AssetManager::getQuadMesh();
    auto shader               = _renderer->getShaderManager()->getShader("edge");
    GraphicsPipeline pipeline = GraphicsPipeline()
                                    .setShader(shader)
                                    .setPrimitiveTopology(PrimitiveTopology::ATCG_POINTS)
                                    .setRasterizerState(RasterizerState().enableCulling(false).setLineSize(2.0f));

    uint32_t entity_id = entity.entity_handle();
    shader->setInt("entityID", entity_id);
    atcg::CameraComponent& comp = entity.getComponent<CameraComponent>();
    shader->setVec3("flat_color", comp.color);
    atcg::ref_ptr<PerspectiveCamera> cam = std::dynamic_pointer_cast<PerspectiveCamera>(comp.camera);
    float aspect_ratio                   = cam->getAspectRatio();
    glm::mat4 scale = glm::scale(glm::vec3(aspect_ratio, 1.0f, -0.5f / glm::tan(glm::radians(cam->getFOV()) / 2.0f)) *
                                 comp.render_scale);
    glm::mat4 model = glm::inverse(cam->getView()) * scale;

    auto points = camera_frustum->getVerticesBuffer();
    GraphicsCommand::bindStorageBuffer(0, points);

    _renderer->drawVAO(camera_frustum->getEdgesArray(), camera, model, pipeline, camera_frustum->n_edges());


    if(comp.image())
    {
        uint32_t id = _renderer->popTextureID();

        auto shader = _renderer->getShaderManager()->getShader("image_display");

        GraphicsPipeline pipeline = GraphicsPipeline()
                                        .setShader(shader)
                                        .setPrimitiveTopology(PrimitiveTopology::ATCG_TRIANGLES)
                                        .setRasterizerState(RasterizerState().enableCulling(false));

        model = model * glm::translate(glm::vec3(0, 0, 1)) * glm::scale(glm::vec3(0.5));
        shader->setInt("screen_texture", id);
        shader->setInt("entityID", entity_id);
        GraphicsCommand::bindTexture(id, comp.image());
        _renderer->drawVAO(quad->getVerticesArray(), camera, model, pipeline, quad->n_vertices());
        _renderer->pushTextureID(id);
    }
    else if(comp.render_preview && comp.preview)
    {
        auto shader = _renderer->getShaderManager()->getShader("image_display");

        GraphicsPipeline pipeline = GraphicsPipeline()
                                        .setShader(shader)
                                        .setPrimitiveTopology(PrimitiveTopology::ATCG_TRIANGLES)
                                        .setRasterizerState(RasterizerState().enableCulling(false));

        uint32_t id = _renderer->popTextureID();
        model       = model * glm::translate(glm::vec3(0, 0, 1)) * glm::scale(glm::vec3(0.5));
        shader->setInt("screen_texture", id);
        shader->setInt("entityID", entity_id);

        GraphicsCommand::bindTexture(id, comp.preview->getColorAttachement(0));
        _renderer->drawVAO(quad->getVerticesArray(), camera, model, pipeline, quad->n_vertices());
        _renderer->pushTextureID(id);
    }
}

namespace GUI
{
void ComponentGUIRenderer<CameraComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                           Entity entity,
                                                           CameraComponent& _component) const
{
#ifndef ATCG_HEADLESS
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

        _component.preview = component.preview;    // This should not count as an update, this is just a lazy init
        // updated = true;
    }

    atcg::Dictionary context;
    context.setValue("camera", component.camera);
    context.setValue("target", component.preview);
    context.setValue("draw_cameras", false);
    scene->draw(context);

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

        atcg::Utils::screenshot(scene, component.camera, component.width, component.height, oss.str());
    }

    ImGui::Separator();

    auto new_handle = displayTexture2DSelection("camera", component.image_handle);

    updated                = (new_handle != component.image_handle) || updated;
    component.image_handle = new_handle;

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
        ATCG_DEBUG("Updated");
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
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT(CameraComponent);
}    // namespace atcg