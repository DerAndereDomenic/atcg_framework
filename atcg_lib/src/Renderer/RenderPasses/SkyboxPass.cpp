#include <Renderer/RenderPasses/SkyboxPass.h>

#include <Renderer/Renderer.h>

namespace atcg
{
SkyboxPass::SkyboxPass(const atcg::ref_ptr<Skybox>& skybox) : RenderPass("SkyboxPass")
{
    _data.setValue("skybox", skybox);
    registerOutput("framebuffer", nullptr);
    setRenderFunction(
        [](Dictionary& context, const Dictionary&, Dictionary& data, Dictionary&)
        {
            bool has_skybox = context.getValueOr<bool>("has_skybox", false);
            auto _skybox    = data.getValue<atcg::ref_ptr<atcg::Skybox>>("skybox");

            if(has_skybox && _skybox)
            {
                atcg::Renderer::drawSkybox(_skybox->getSkyboxCubeMap(),
                                           context.getValue<atcg::ref_ptr<Camera>>("camera"));
            }
        });
}
}    // namespace atcg