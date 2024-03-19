#include <Renderer/VRRenderer.h>

namespace atcg
{
VRRenderer* VRRenderer::s_renderer = new VRRenderer;

class VRRenderer::Impl
{
public:
    Impl();

    ~Impl() = default;
};

VRRenderer::VRRenderer() {}

VRRenderer::~VRRenderer() {}

VRRenderer::Impl::Impl() {}

void VRRenderer::init(const EventCallbackFn& callback)
{
    s_renderer->impl = atcg::make_scope<Impl>();
}

void VRRenderer::onUpdate(const float delta_time) {}

glm::mat4 VRRenderer::getView(const Eye& eye)
{
    return glm::mat4(1);
}

std::tuple<glm::mat4, glm::mat4> VRRenderer::getViews()
{
    return std::make_tuple(VRRenderer::getView(VRRenderer::Eye::LEFT), VRRenderer::getView(VRRenderer::Eye::RIGHT));
}

glm::mat4 VRRenderer::getProjection(const Eye& eye)
{
    return glm::mat4(1);
}

std::tuple<glm::mat4, glm::mat4> VRRenderer::getProjections()
{
    return std::make_tuple(VRRenderer::getProjection(VRRenderer::Eye::LEFT),
                           VRRenderer::getProjection(VRRenderer::Eye::RIGHT));
}

atcg::ref_ptr<Framebuffer> VRRenderer::getRenderTarget(const Eye& eye)
{
    return nullptr;
}

std::tuple<atcg::ref_ptr<Framebuffer>, atcg::ref_ptr<Framebuffer>> VRRenderer::getRenderTargets()
{
    return std::make_tuple(VRRenderer::getRenderTarget(VRRenderer::Eye::LEFT),
                           VRRenderer::getRenderTarget(VRRenderer::Eye::RIGHT));
}

glm::vec3 VRRenderer::getPosition()
{
    return glm::vec3(0);
}

uint32_t VRRenderer::width()
{
    return 0;
}

uint32_t VRRenderer::height()
{
    return 0;
}
}    // namespace atcg