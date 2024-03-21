#include <Renderer/VRRenderer.h>

#include <Core/Log.h>
#include <glad/glad.h>
#include <DataStructure/Graph.h>
#include <Renderer/Renderer.h>
#include <Events/VREvent.h>
#include <openvr.h>

namespace atcg
{
VRRenderer* VRRenderer::s_renderer = new VRRenderer;

class VRRenderer::Impl
{
public:
    Impl();

    ~Impl() = default;

    void init();
    void deinit();
    bool vr_available = false;
    EventCallbackFn on_event;

    vr::IVRSystem* vr_pointer = NULL;

    uint32_t width, height;
    atcg::ref_ptr<Framebuffer> render_target_left, render_target_right;
    glm::mat4 projection_left  = glm::mat4(1);
    glm::mat4 projection_right = glm::mat4(1);

    glm::mat4 inv_view_left  = glm::mat4(1);
    glm::mat4 inv_view_right = glm::mat4(1);

    glm::vec3 position = glm::vec3(0);

    atcg::ref_ptr<Graph> quad;
};

VRRenderer::VRRenderer() {}

VRRenderer::~VRRenderer()
{
    impl->deinit();
}

VRRenderer::Impl::Impl() {}

void VRRenderer::Impl::init()
{
    bool hmd_present       = vr::VR_IsHmdPresent();
    bool runtime_installed = vr::VR_IsRuntimeInstalled();

    if(hmd_present && runtime_installed)
    {
        vr::EVRInitError error = vr::VRInitError_None;
        vr_pointer             = vr::VR_Init(&error, vr::VRApplication_Scene);
        if(error != vr::VRInitError_None)
        {
            vr_pointer = NULL;
            ATCG_WARN("Unable to init VR runtime: {0}", std::string(vr::VR_GetVRInitErrorAsEnglishDescription(error)));
        }
        vr_available = true;
    }

    vr_pointer->GetRecommendedRenderTargetSize(&width, &height);

    render_target_left = atcg::make_ref<Framebuffer>(width, height);
    render_target_left->attachColor();
    render_target_left->attachDepth();
    render_target_left->complete();

    render_target_right = atcg::make_ref<Framebuffer>(width, height);
    render_target_right->attachColor();
    render_target_right->attachDepth();
    render_target_right->complete();

    std::vector<atcg::Vertex> vertices = {atcg::Vertex(glm::vec3(-1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, 1, 0)),
                                          atcg::Vertex(glm::vec3(-1, 1, 0))};

    std::vector<glm::u32vec3> edges = {glm::u32vec3(0, 1, 2), glm::u32vec3(0, 2, 3)};

    quad = atcg::Graph::createTriangleMesh(vertices, edges);

    ATCG_INFO("Initialized VR runtime with resolution: {0}x{1}", width, height);
}

void VRRenderer::Impl::deinit()
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

void VRRenderer::init(const EventCallbackFn& callback)
{
    s_renderer->impl = atcg::make_scope<Impl>();

    s_renderer->impl->init();
    s_renderer->impl->on_event = callback;

    doTracking();
}

void VRRenderer::onUpdate(const float delta_time)
{
    // Upload to HMD
    {
        vr::Texture_t left_eye_texture  = {(void*)s_renderer->impl->render_target_left->getColorAttachement()->getID(),
                                           vr::TextureType_OpenGL,
                                           vr::ColorSpace::ColorSpace_Linear};
        vr::Texture_t right_eye_texture = {(void*)s_renderer->impl->render_target_right->getColorAttachement()->getID(),
                                           vr::TextureType_OpenGL,
                                           vr::ColorSpace::ColorSpace_Linear};

        vr::VRCompositor()->Submit(vr::Eye_Left, &left_eye_texture);
        vr::VRCompositor()->Submit(vr::Eye_Right, &right_eye_texture);

        glFlush();

        // vr::VRCompositor()->PostPresentHandoff();
    }
}

void VRRenderer::doTracking()
{
    vr::TrackedDevicePose_t renderPoses[vr::k_unMaxTrackedDeviceCount];

    vr::VRCompositor()->WaitGetPoses(renderPoses, vr::k_unMaxTrackedDeviceCount, nullptr, 0);

    // Update position
    vr::TrackedDevicePose_t trackedDevicePose = renderPoses[vr::k_unTrackedDeviceIndex_Hmd];

    {
        s_renderer->impl->position = glm::vec3(trackedDevicePose.mDeviceToAbsoluteTracking.m[0][3],
                                               trackedDevicePose.mDeviceToAbsoluteTracking.m[1][3],
                                               trackedDevicePose.mDeviceToAbsoluteTracking.m[2][3]);
    }

    // Update view matrices
    {
        vr::HmdMatrix34_t e = s_renderer->impl->vr_pointer->GetEyeToHeadTransform(vr::EVREye::Eye_Left);
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
        s_renderer->impl->inv_view_left = result * eye2head;
    }

    {
        vr::HmdMatrix34_t e = s_renderer->impl->vr_pointer->GetEyeToHeadTransform(vr::EVREye::Eye_Right);
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
        s_renderer->impl->inv_view_right = result * eye2head;
    }

    // Update projection matrices
    {
        vr::HmdMatrix44_t projection =
            s_renderer->impl->vr_pointer->GetProjectionMatrix(vr::EVREye::Eye_Left, 0.01f, 1000.0f);
        glm::mat4 result = glm::mat4(1);
        for(int i = 0; i < 4; ++i)
        {
            for(int j = 0; j < 4; ++j)
            {
                result[i][j] = projection.m[j][i];
            }
        }

        s_renderer->impl->projection_left = result;
    }

    {
        vr::HmdMatrix44_t projection =
            s_renderer->impl->vr_pointer->GetProjectionMatrix(vr::EVREye::Eye_Right, 0.01f, 1000.0f);
        glm::mat4 result = glm::mat4(1);
        for(int i = 0; i < 4; ++i)
        {
            for(int j = 0; j < 4; ++j)
            {
                result[i][j] = projection.m[j][i];
            }
        }

        s_renderer->impl->projection_right = result;
    }
}

void VRRenderer::emitEvents()
{
    vr::VREvent_t vrevent;
    if(s_renderer->impl->vr_pointer->PollNextEvent(&vrevent, sizeof(vrevent)))
    {
        if(vrevent.eventType == vr::EVREventType::VREvent_ButtonPress)
        {
            VRButtonPressedEvent event(vrevent.data.controller.button, vrevent.trackedDeviceIndex);
            s_renderer->impl->on_event(&event);
        }
        else if(vrevent.eventType == vr::EVREventType::VREvent_ButtonUnpress)
        {
            VRButtonReleasedEvent event(vrevent.data.controller.button, vrevent.trackedDeviceIndex);
            s_renderer->impl->on_event(&event);
        }
        else if(vrevent.eventType == vr::EVREventType::VREvent_ButtonTouch)
        {
            VRButtonTouchedEvent event(vrevent.data.controller.button, vrevent.trackedDeviceIndex);
            s_renderer->impl->on_event(&event);
        }
        else if(vrevent.eventType == vr::EVREventType::VREvent_ButtonUntouch)
        {
            VRButtonUntouchedEvent event(vrevent.data.controller.button, vrevent.trackedDeviceIndex);
            s_renderer->impl->on_event(&event);
        }
    }
}

glm::mat4 VRRenderer::getInverseView(const Eye& eye)
{
    if(eye == Eye::LEFT)
    {
        return s_renderer->impl->inv_view_left;
    }
    return s_renderer->impl->inv_view_right;
}

std::tuple<glm::mat4, glm::mat4> VRRenderer::getInverseViews()
{
    return std::make_tuple(VRRenderer::getInverseView(VRRenderer::Eye::LEFT),
                           VRRenderer::getInverseView(VRRenderer::Eye::RIGHT));
}

glm::mat4 VRRenderer::getProjection(const Eye& eye)
{
    if(eye == Eye::LEFT)
    {
        return s_renderer->impl->projection_left;
    }
    return s_renderer->impl->projection_right;
}

std::tuple<glm::mat4, glm::mat4> VRRenderer::getProjections()
{
    return std::make_tuple(VRRenderer::getProjection(VRRenderer::Eye::LEFT),
                           VRRenderer::getProjection(VRRenderer::Eye::RIGHT));
}

atcg::ref_ptr<Framebuffer> VRRenderer::getRenderTarget(const Eye& eye)
{
    if(eye == Eye::LEFT)
    {
        return s_renderer->impl->render_target_left;
    }
    return s_renderer->impl->render_target_right;
}

std::tuple<atcg::ref_ptr<Framebuffer>, atcg::ref_ptr<Framebuffer>> VRRenderer::getRenderTargets()
{
    return std::make_tuple(VRRenderer::getRenderTarget(VRRenderer::Eye::LEFT),
                           VRRenderer::getRenderTarget(VRRenderer::Eye::RIGHT));
}

void VRRenderer::renderToScreen()
{
    auto vr_shader = atcg::ShaderManager::getShader("vrScreen");
    vr_shader->setInt("texture_left", 10);
    vr_shader->setInt("texture_right", 11);
    s_renderer->impl->render_target_left->getColorAttachement()->use(10);
    s_renderer->impl->render_target_right->getColorAttachement()->use(11);
    atcg::Renderer::draw(s_renderer->impl->quad, {}, glm::mat4(1), glm::vec3(1), vr_shader);
}

glm::vec3 VRRenderer::getPosition()
{
    return s_renderer->impl->position;
}

bool VRRenderer::isVRAvailable()
{
    return s_renderer->impl->vr_available;
}

uint32_t VRRenderer::width()
{
    return s_renderer->impl->width;
}

uint32_t VRRenderer::height()
{
    return s_renderer->impl->height;
}
}    // namespace atcg