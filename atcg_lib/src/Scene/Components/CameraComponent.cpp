#include <Scene/Components/CameraComponent.h>
#include <Scene/ComponentRegistry.h>
#include <Scene/SceneRenderer.h>

#include <Core/Application.h>
#include <Utils/Utils.h>

#define PERSPECTIVE_CAMERA_KEY "PerspectiveCamera"
#define CAMERA_IMAGE_KEY       "Image"
#define ASPECT_RATIO_KEY       "AspectRatio"
#define FOVY_KEY               "FoVy"
#define LOOKAT_KEY             "LookAt"
#define NEAR_KEY               "Near"
#define FAR_KEY                "Far"
#define WIDTH_KEY              "width"
#define HEIGHT_KEY             "height"
#define PREVIEW_KEY            "preview"
#define OPTICAL_CENTER_KEY     "OpticalCenter"
#define RENDER_SCALE_KEY       "Scale"
#define POSITION_KEY           "Position"

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

namespace Serialization
{
void ComponentSerializer<CameraComponent>::serialize_component(const std::string& file_path,
                                                               const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               CameraComponent& component,
                                                               nlohmann::json& j) const
{
    atcg::ref_ptr<PerspectiveCamera> cam = std::dynamic_pointer_cast<PerspectiveCamera>(component.camera);

    glm::vec3 position = cam->getPosition();
    glm::vec3 look_at  = cam->getLookAt();
    glm::vec2 offset   = cam->getIntrinsics().opticalCenter();
    float n            = cam->getNear();
    float f            = cam->getFar();

    j[PERSPECTIVE_CAMERA_KEY][ASPECT_RATIO_KEY]   = cam->getAspectRatio();
    j[PERSPECTIVE_CAMERA_KEY][FOVY_KEY]           = cam->getFOV();
    j[PERSPECTIVE_CAMERA_KEY][POSITION_KEY]       = nlohmann::json::array({position.x, position.y, position.z});
    j[PERSPECTIVE_CAMERA_KEY][LOOKAT_KEY]         = nlohmann::json::array({look_at.x, look_at.y, look_at.z});
    j[PERSPECTIVE_CAMERA_KEY][NEAR_KEY]           = n;
    j[PERSPECTIVE_CAMERA_KEY][FAR_KEY]            = f;
    j[PERSPECTIVE_CAMERA_KEY][WIDTH_KEY]          = component.width;
    j[PERSPECTIVE_CAMERA_KEY][HEIGHT_KEY]         = component.height;
    j[PERSPECTIVE_CAMERA_KEY][PREVIEW_KEY]        = component.render_preview;
    j[PERSPECTIVE_CAMERA_KEY][OPTICAL_CENTER_KEY] = nlohmann::json::array({offset.x, offset.y});
    j[PERSPECTIVE_CAMERA_KEY][RENDER_SCALE_KEY]   = component.render_scale;

    if(component.image())
    {
        j[PERSPECTIVE_CAMERA_KEY][CAMERA_IMAGE_KEY] = (uint64_t)component.image_handle;
    }
}

void ComponentSerializer<CameraComponent>::deserialize_component(const std::string& file_path,
                                                                 const atcg::ref_ptr<Scene>& scene,
                                                                 Entity entity,
                                                                 nlohmann::json& j) const
{
    if(!j.contains(PERSPECTIVE_CAMERA_KEY))
    {
        return;
    }

    float aspect_ratio          = j[PERSPECTIVE_CAMERA_KEY].value(ASPECT_RATIO_KEY, 1.0f);
    float fov                   = j[PERSPECTIVE_CAMERA_KEY].value(FOVY_KEY, 60.0f);
    float n                     = j[PERSPECTIVE_CAMERA_KEY].value(NEAR_KEY, 0.01f);
    float f                     = j[PERSPECTIVE_CAMERA_KEY].value(FAR_KEY, 1000.0f);
    std::vector<float> position = j[PERSPECTIVE_CAMERA_KEY].value(POSITION_KEY, std::vector<float> {0.0f, 0.0f, -1.0f});
    std::vector<float> lookat   = j[PERSPECTIVE_CAMERA_KEY].value(LOOKAT_KEY, std::vector<float> {0.0f, 0.0f, 0.0f});
    std::vector<float> offset   = j[PERSPECTIVE_CAMERA_KEY].value(OPTICAL_CENTER_KEY, std::vector<float> {0.0f, 0.0f});

    CameraExtrinsics extrinsics(glm::make_vec3(position.data()), glm::make_vec3(lookat.data()));
    CameraIntrinsics intrinsics(aspect_ratio, fov, n, f);
    intrinsics.setOpticalCenter(glm::make_vec2(offset.data()));

    auto cam = atcg::make_ref<atcg::PerspectiveCamera>(extrinsics, intrinsics);

    auto& component          = entity.addComponent<CameraComponent>(cam);
    component.width          = j[PERSPECTIVE_CAMERA_KEY].value(WIDTH_KEY, 1024);
    component.height         = j[PERSPECTIVE_CAMERA_KEY].value(HEIGHT_KEY, 1024);
    component.render_preview = j[PERSPECTIVE_CAMERA_KEY].value(PREVIEW_KEY, false);
    component.render_scale   = j[PERSPECTIVE_CAMERA_KEY].value(RENDER_SCALE_KEY, 1.0f);


    if(j[PERSPECTIVE_CAMERA_KEY].contains(CAMERA_IMAGE_KEY))
    {
        component.image_handle = (AssetHandle)j[PERSPECTIVE_CAMERA_KEY][CAMERA_IMAGE_KEY];
    }
}

}    // namespace Serialization
namespace GUI
{
void ComponentGUIRenderer<CameraComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                           Entity entity,
                                                           CameraComponent& _component) const
{
#ifndef ATCG_HEADLESS
    CameraComponent component = _component;
    bool updated              = false;
    bool deactivated          = false;

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
    deactivated = ImGui::IsItemDeactivated() || deactivated;

    label.str(std::string());
    label << "FOV##" << id;
    if(ImGui::DragFloat(label.str().c_str(), &fov, 0.5f, 10.0f, 120.0f))
    {
        intrinsics.setFOV(fov);
        updated = true;
    }
    deactivated = ImGui::IsItemDeactivated() || deactivated;

    label.str(std::string());
    label << "Resolution##" << id;
    if(ImGui::DragInt2(label.str().c_str(), (int*)res, 1, 1, 4096))
    {
        component.width  = res[0];
        component.height = res[1];
        updated          = true;
    }
    deactivated = ImGui::IsItemDeactivated() || deactivated;

    label.str(std::string());
    label << "Optical Center##" << id;
    if(ImGui::DragFloat2(label.str().c_str(), offset, 0.01f, -1.0f, 1.0f))
    {
        intrinsics.setOpticalCenter(glm::make_vec2(offset));
        updated = true;
    }
    deactivated = ImGui::IsItemDeactivated() || deactivated;

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
    deactivated = ImGui::IsItemDeactivated() || deactivated;

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
    deactivated = ImGui::IsItemDeactivated() || deactivated;

    label.str(std::string());
    label << "Color##" << id;
    updated     = ImGui::ColorEdit3(label.str().c_str(), glm::value_ptr(component.color)) || updated;
    deactivated = ImGui::IsItemDeactivated() || deactivated;

    label.str(std::string());
    label << "Scale##" << id;
    updated     = ImGui::DragFloat(label.str().c_str(), &component.render_scale, 0.01f, 0.01f, FLT_MAX) || updated;
    deactivated = ImGui::IsItemDeactivated() || deactivated;

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

    atcg::SceneRenderer::render(scene, component.camera, component.preview, false);

    updated     = ImGui::Checkbox("Show Preview##cam", &component.render_preview) || updated;
    deactivated = ImGui::IsItemDeactivated() || deactivated;

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

    auto new_handle = Utils::displayTexture2DSelection("camera", component.image_handle, deactivated);

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
        deactivated = ImGui::IsItemDeactivated() || deactivated;
    }

    if(updated && !atcg::RevisionStack::isRecording())
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<CameraComponent>>(scene, entity);
    }

    if(updated)
    {
        _component = component;
        camera->setIntrinsics(intrinsics);
        _component.camera = camera;
    }

    if(deactivated && atcg::RevisionStack::isRecording())
    {
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