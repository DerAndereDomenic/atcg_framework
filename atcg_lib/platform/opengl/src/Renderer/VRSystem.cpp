#include <Renderer/VRSystem.h>

#include <Core/Log.h>
#include <Core/Path.h>
#include <glad/glad.h>
#include <DataStructure/Graph.h>
#include <Renderer/Renderer.h>
#include <Events/VREvent.h>
#include <openvr.h>

#include <Scene/Entity.h>
#include <Scene/Components.h>

namespace atcg
{

class VRSystem::Impl
{
public:
    Impl();

    ~Impl() = default;

    void init();
    void deinit();
    bool vr_available = false;
    EventCallbackFn on_event;

    vr::IVRSystem* vr_pointer = NULL;

    uint32_t width = 0, height = 0;
    atcg::ref_ptr<Framebuffer> render_target_left, render_target_right;
    glm::mat4 projection_left  = glm::mat4(1);
    glm::mat4 projection_right = glm::mat4(1);

    glm::mat4 inv_view_left  = glm::mat4(1);
    glm::mat4 inv_view_right = glm::mat4(1);

    glm::vec3 position = glm::vec3(0);

    atcg::ref_ptr<Graph> quad;
    atcg::ref_ptr<Graph> movement_line;

    // Tracked poses
    vr::TrackedDevicePose_t renderPoses[vr::k_unMaxTrackedDeviceCount];

    bool controller_initialized = false;
    atcg::Entity left_controller_entity;
    atcg::Entity right_controller_entity;

    glm::vec3 offset = glm::vec3(0);

    float n = 0.01f;
    float f = 1000.0f;
};

VRSystem::VRSystem() {}

VRSystem::~VRSystem()
{
    impl->deinit();
}

VRSystem::Impl::Impl() {}

void VRSystem::Impl::init()
{
    bool hmd_present       = vr::VR_IsHmdPresent();
    bool runtime_installed = vr::VR_IsRuntimeInstalled();

    if(hmd_present && runtime_installed)
    {
        vr::EVRInitError error = vr::VRInitError_None;
        vr_pointer             = vr::VR_Init(&error, vr::VRApplication_Scene);

        if(error != vr::VRInitError_None || vr_pointer == NULL)
        {
            vr_pointer = NULL;
            ATCG_WARN("Unable to init VR runtime: {0}", std::string(vr::VR_GetVRInitErrorAsEnglishDescription(error)));
            return;
        }
        vr_available = true;
        vr_pointer->GetRecommendedRenderTargetSize(&width, &height);


        render_target_left = atcg::make_ref<Framebuffer>(width, height);
        render_target_left->attachColor();
        render_target_left->attachDepth();
        render_target_left->complete();

        render_target_right = atcg::make_ref<Framebuffer>(width, height);
        render_target_right->attachColor();
        render_target_right->attachDepth();
        render_target_right->complete();

        {
            std::vector<atcg::Vertex> vertices = {atcg::Vertex(glm::vec3(-1, -1, 0)),
                                                  atcg::Vertex(glm::vec3(1, -1, 0)),
                                                  atcg::Vertex(glm::vec3(1, 1, 0)),
                                                  atcg::Vertex(glm::vec3(-1, 1, 0))};

            std::vector<glm::u32vec3> edges = {glm::u32vec3(0, 1, 2), glm::u32vec3(0, 2, 3)};

            quad = atcg::Graph::createTriangleMesh(vertices, edges);
        }

        {
            std::vector<atcg::Vertex> vertices = {atcg::Vertex(glm::vec3(0)), atcg::Vertex(glm::vec3(0))};

            std::vector<atcg::Edge> edges = {atcg::Edge {glm::vec2(0, 1), glm::vec3(1), 1.0f}};

            movement_line = atcg::Graph::createGraph(vertices, edges);
        }

        ATCG_INFO("Initialized VR runtime with resolution: {0}x{1}", width, height);
    }
}

void VRSystem::Impl::deinit()
{
    if(vr_pointer != NULL)
    {
        vr::VR_Shutdown();
        vr_pointer = NULL;
    }
    width        = 0;
    height       = 0;
    vr_available = false;
}

void VRSystem::init(const EventCallbackFn& callback)
{
    impl = atcg::make_scope<Impl>();

    impl->init();
    impl->on_event = callback;

    doTracking();
}

void VRSystem::initControllerMeshes(const atcg::ref_ptr<atcg::Scene>& scene)
{
    if(!impl->vr_available) return;

    // Left
    {
        auto mesh = atcg::IO::read_mesh(
            (atcg::resource_directory() / "VRController/Quest/questpro_controllers_left.obj").string());

        auto base_color =
            atcg::IO::imread((atcg::resource_directory() / "VRController/Quest/controller_l_lo_BaseColor.png").string(),
                             2.2f);
        auto normal =
            atcg::IO::imread((atcg::resource_directory() / "VRController/Quest/controller_l_lo_Normal.png").string());
        auto roughness = atcg::IO::imread(
            (atcg::resource_directory() / "VRController/Quest/controller_l_lo_roughness.png").string());
        auto metallic =
            atcg::IO::imread((atcg::resource_directory() / "VRController/Quest/controller_l_lo_metallic.png").string());

        impl->left_controller_entity = scene->createEntity("Left "
                                                           "Controll"
                                                           "er");
        impl->left_controller_entity.addComponent<atcg::TransformComponent>();
        auto& renderer = impl->left_controller_entity.addComponent<atcg::MeshRenderComponent>();
        renderer.material()->setDiffuseTexture(atcg::Texture2D::create(base_color));
        renderer.material()->setNormalTexture(atcg::Texture2D::create(normal));
        renderer.material()->setRoughnessTexture(atcg::Texture2D::create(roughness));
        renderer.material()->setMetallicTexture(atcg::Texture2D::create(metallic));

        impl->left_controller_entity.addComponent<atcg::GeometryComponent>(mesh);
    }

    // Right
    {
        auto mesh = atcg::IO::read_mesh(
            (atcg::resource_directory() / "VRController/Quest/questpro_controllers_right.obj").string());

        auto base_color =
            atcg::IO::imread((atcg::resource_directory() / "VRController/Quest/controller_r_lo_BaseColor.png").string(),
                             2.2f);
        auto normal =
            atcg::IO::imread((atcg::resource_directory() / "VRController/Quest/controller_r_lo_Normal.png").string());
        auto roughness = atcg::IO::imread(
            (atcg::resource_directory() / "VRController/Quest/controller_r_lo_roughness.png").string());
        auto metallic =
            atcg::IO::imread((atcg::resource_directory() / "VRController/Quest/controller_r_lo_metallic.png").string());

        impl->right_controller_entity = scene->createEntity("Right "
                                                            "Control"
                                                            "ler");
        impl->right_controller_entity.addComponent<atcg::TransformComponent>();
        auto& renderer = impl->right_controller_entity.addComponent<atcg::MeshRenderComponent>();
        renderer.material()->setDiffuseTexture(atcg::Texture2D::create(base_color));
        renderer.material()->setNormalTexture(atcg::Texture2D::create(normal));
        renderer.material()->setRoughnessTexture(atcg::Texture2D::create(roughness));
        renderer.material()->setMetallicTexture(atcg::Texture2D::create(metallic));
        impl->right_controller_entity.addComponent<atcg::GeometryComponent>(mesh);
    }

    impl->controller_initialized = true;
}

void VRSystem::onUpdate(const float delta_time)
{
    if(!impl->vr_available) return;

    // Upload to HMD
    {
        vr::Texture_t left_eye_texture  = {(void*)(uint64_t*)impl->render_target_left->getColorAttachement()->getID(),
                                           vr::TextureType_OpenGL,
                                           vr::ColorSpace::ColorSpace_Linear};
        vr::Texture_t right_eye_texture = {(void*)(uint64_t*)impl->render_target_right->getColorAttachement()->getID(),
                                           vr::TextureType_OpenGL,
                                           vr::ColorSpace::ColorSpace_Linear};

        vr::VRCompositor()->Submit(vr::Eye_Left, &left_eye_texture);
        vr::VRCompositor()->Submit(vr::Eye_Right, &right_eye_texture);

        glFlush();

        // vr::VRCompositor()->PostPresentHandoff();
    }

    // Update controller transforms
    if(impl->controller_initialized)
    {
        {
            uint32_t device_idx = impl->vr_pointer->GetTrackedDeviceIndexForControllerRole(
                vr::ETrackedControllerRole::TrackedControllerRole_RightHand);

            if(device_idx >= vr::k_unMaxTrackedDeviceCount) return;

            glm::mat4 model = VRSystem::getDevicePose(device_idx);

            impl->right_controller_entity.getComponent<atcg::TransformComponent>().setModel(model);
        }

        {
            uint32_t device_idx = impl->vr_pointer->GetTrackedDeviceIndexForControllerRole(
                vr::ETrackedControllerRole::TrackedControllerRole_LeftHand);

            if(device_idx >= vr::k_unMaxTrackedDeviceCount) return;

            glm::mat4 model = VRSystem::getDevicePose(device_idx);

            impl->left_controller_entity.getComponent<atcg::TransformComponent>().setModel(model);
        }
    }
}

void VRSystem::doTracking()
{
    if(!impl->vr_available) return;

    vr::VRCompositor()->WaitGetPoses(impl->renderPoses, vr::k_unMaxTrackedDeviceCount, nullptr, 0);

    // Update position
    vr::TrackedDevicePose_t trackedDevicePose = impl->renderPoses[vr::k_unTrackedDeviceIndex_Hmd];

    {
        impl->position = glm::vec3(trackedDevicePose.mDeviceToAbsoluteTracking.m[0][3],
                                   trackedDevicePose.mDeviceToAbsoluteTracking.m[1][3],
                                   trackedDevicePose.mDeviceToAbsoluteTracking.m[2][3]);
    }

    // Update view matrices
    {
        vr::HmdMatrix34_t e = impl->vr_pointer->GetEyeToHeadTransform(vr::EVREye::Eye_Left);
        glm::mat4 result    = glm::mat4(1);
        glm::mat4 eye2head  = glm::mat4(1);
        if(trackedDevicePose.bPoseIsValid)
        {
            for(int i = 0; i < 3; ++i)
            {
                for(int j = 0; j < 4; ++j)
                {
                    result[j][i]   = trackedDevicePose.mDeviceToAbsoluteTracking.m[i][j];
                    eye2head[j][i] = e.m[i][j];
                }
            }
        }

        result = result * eye2head;
        result[3] += glm::vec4(impl->offset, 0);

        impl->inv_view_left = result;
    }

    {
        vr::HmdMatrix34_t e = impl->vr_pointer->GetEyeToHeadTransform(vr::EVREye::Eye_Right);
        glm::mat4 result    = glm::mat4(1);
        glm::mat4 eye2head  = glm::mat4(1);
        if(trackedDevicePose.bPoseIsValid)
        {
            for(int i = 0; i < 3; ++i)
            {
                for(int j = 0; j < 4; ++j)
                {
                    result[j][i]   = trackedDevicePose.mDeviceToAbsoluteTracking.m[i][j];
                    eye2head[j][i] = e.m[i][j];
                }
            }
        }

        result = result * eye2head;
        result[3] += glm::vec4(impl->offset, 0);

        impl->inv_view_right = result;
    }

    // Update projection matrices
    {
        vr::HmdMatrix44_t projection = impl->vr_pointer->GetProjectionMatrix(
            vr::EVREye::Eye_Left,
            2.0f * impl->n,
            impl->f);    // ? SteamVR Bug, need to pass 2*near to get the correct value in the projection matrix
        glm::mat4 result = glm::mat4(1);
        for(int i = 0; i < 4; ++i)
        {
            for(int j = 0; j < 4; ++j)
            {
                result[i][j] = projection.m[j][i];
            }
        }

        impl->projection_left = result;
    }

    {
        vr::HmdMatrix44_t projection = impl->vr_pointer->GetProjectionMatrix(
            vr::EVREye::Eye_Right,
            2.0f * impl->n,
            impl->f);    // ? SteamVR Bug, need to pass 2*near to get the correct value in the projection matrix
        glm::mat4 result = glm::mat4(1);
        for(int i = 0; i < 4; ++i)
        {
            for(int j = 0; j < 4; ++j)
            {
                result[i][j] = projection.m[j][i];
            }
        }

        impl->projection_right = result;
    }
}

void VRSystem::emitEvents()
{
    if(!impl->vr_available) return;

    vr::VREvent_t vrevent;
    if(impl->vr_pointer->PollNextEvent(&vrevent, sizeof(vrevent)))
    {
        if(vrevent.eventType == vr::EVREventType::VREvent_ButtonPress)
        {
            VRButtonPressedEvent event(vrevent.data.controller.button, vrevent.trackedDeviceIndex);
            impl->on_event(&event);
        }
        else if(vrevent.eventType == vr::EVREventType::VREvent_ButtonUnpress)
        {
            VRButtonReleasedEvent event(vrevent.data.controller.button, vrevent.trackedDeviceIndex);
            impl->on_event(&event);
        }
        else if(vrevent.eventType == vr::EVREventType::VREvent_ButtonTouch)
        {
            VRButtonTouchedEvent event(vrevent.data.controller.button, vrevent.trackedDeviceIndex);
            impl->on_event(&event);
        }
        else if(vrevent.eventType == vr::EVREventType::VREvent_ButtonUntouch)
        {
            VRButtonUntouchedEvent event(vrevent.data.controller.button, vrevent.trackedDeviceIndex);
            impl->on_event(&event);
        }
    }
}

glm::mat4 VRSystem::getInverseView(const Eye& eye)
{
    if(eye == Eye::LEFT)
    {
        return impl->inv_view_left;
    }
    return impl->inv_view_right;
}

std::tuple<glm::mat4, glm::mat4> VRSystem::getInverseViews()
{
    return std::make_tuple(VRSystem::getInverseView(VRSystem::Eye::LEFT),
                           VRSystem::getInverseView(VRSystem::Eye::RIGHT));
}

glm::mat4 VRSystem::getProjection(const Eye& eye)
{
    if(eye == Eye::LEFT)
    {
        return impl->projection_left;
    }
    return impl->projection_right;
}

std::tuple<glm::mat4, glm::mat4> VRSystem::getProjections()
{
    return std::make_tuple(VRSystem::getProjection(VRSystem::Eye::LEFT), VRSystem::getProjection(VRSystem::Eye::RIGHT));
}

atcg::ref_ptr<Framebuffer> VRSystem::getRenderTarget(const Eye& eye)
{
    if(eye == Eye::LEFT)
    {
        return impl->render_target_left;
    }
    return impl->render_target_right;
}

std::tuple<atcg::ref_ptr<Framebuffer>, atcg::ref_ptr<Framebuffer>> VRSystem::getRenderTargets()
{
    return std::make_tuple(VRSystem::getRenderTarget(VRSystem::Eye::LEFT),
                           VRSystem::getRenderTarget(VRSystem::Eye::RIGHT));
}

void VRSystem::renderToScreen()
{
    if(!impl->vr_available) return;

    auto vr_shader = atcg::ShaderManager::getShader("vrScreen");
    vr_shader->setInt("texture_left", 10);
    vr_shader->setInt("texture_right", 11);
    impl->render_target_left->getColorAttachement()->use(10);
    impl->render_target_right->getColorAttachement()->use(11);
    atcg::Renderer::toggleDepthTesting(false);
    atcg::Renderer::draw(impl->quad, {}, glm::mat4(1), glm::vec3(1), vr_shader);
    atcg::Renderer::toggleDepthTesting(true);
}

glm::vec3 VRSystem::getPosition()
{
    return impl->position;
}

bool VRSystem::isVRAvailable()
{
    return impl->vr_available;
}

uint32_t VRSystem::width()
{
    return impl->width;
}

uint32_t VRSystem::height()
{
    return impl->height;
}

VRSystem::Role VRSystem::getDeviceRole(const uint32_t device_index)
{
    if(!impl->vr_available) return VRSystem::Role::INVALID;

    if(device_index == vr::k_unTrackedDeviceIndex_Hmd) return VRSystem::Role::HMD;

    auto device_role = impl->vr_pointer->GetControllerRoleForTrackedDeviceIndex(device_index);

    if(device_role == vr::ETrackedControllerRole::TrackedControllerRole_LeftHand)
    {
        return VRSystem::Role::LEFT_HAND;
    }
    else if(device_role == vr::ETrackedControllerRole::TrackedControllerRole_RightHand)
    {
        return VRSystem::Role::RIGHT_HAND;
    }

    return VRSystem::Role::INVALID;
}

glm::mat4 VRSystem::getDevicePose(const uint32_t device_index)
{
    if(!impl->vr_available) return glm::mat4(1);

    auto trackedDevicePose = impl->renderPoses[device_index];

    glm::mat4 result = glm::mat4(1);
    if(trackedDevicePose.bPoseIsValid)
    {
        for(int i = 0; i < 3; ++i)
        {
            for(int j = 0; j < 4; ++j)
            {
                result[j][i] = trackedDevicePose.mDeviceToAbsoluteTracking.m[i][j];
            }
        }
    }

    result[3] += glm::vec4(impl->offset, 0);

    return result;
}

void VRSystem::setMovementLine(const glm::vec3& start, const glm::vec3& end)
{
    if(!impl->vr_available) return;

    auto positions = impl->movement_line->getDevicePositions();

    torch::Tensor line_tensor =
        torch::tensor({{start.x, start.y, start.z}, {end.x, end.y, end.z}}, atcg::TensorOptions::floatDeviceOptions());

    positions.index_put_({torch::indexing::Slice(), torch::indexing::Slice()}, line_tensor);

    impl->movement_line->unmapAllPointers();
}

void VRSystem::drawMovementLine(const atcg::ref_ptr<atcg::PerspectiveCamera>& camera)
{
    if(!impl->vr_available) return;

    atcg::ShaderManager::getShader("edge")->setFloat("fall_off_edge", 1000.0f);
    atcg::ShaderManager::getShader("edge")->setFloat("base_transparency", 1.0f);
    atcg::Renderer::draw(impl->movement_line,
                         camera,
                         glm::mat4(1),
                         glm::vec3(1),
                         nullptr,
                         atcg::DrawMode::ATCG_DRAW_MODE_EDGES);
}

void VRSystem::setOffset(const glm::vec3& offset)
{
    impl->offset = offset;
}

glm::vec3 VRSystem::getOffset()
{
    return impl->offset;
}

void VRSystem::setNear(const float n)
{
    impl->n = n;
}

void VRSystem::setFar(const float f)
{
    impl->f = f;
}

}    // namespace atcg