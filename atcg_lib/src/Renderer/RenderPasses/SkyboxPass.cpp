#include <Renderer/RenderPasses/SkyboxPass.h>

#include <Renderer/Renderer.h>

namespace atcg
{
SkyboxPass::SkyboxPass() : RenderPass("SkyboxPass")
{
    registerOutput("framebuffer", nullptr);
    setRenderFunction([](Dictionary& context, const Dictionary&, Dictionary&, Dictionary&)
                      { atcg::Renderer::drawSkybox(context.getValue<atcg::ref_ptr<Camera>>("camera")); });
}
}    // namespace atcg