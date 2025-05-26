#include <Renderer/RenderPasses/SkyboxPass.h>

#include <Renderer/Renderer.h>

namespace atcg
{
SkyboxPass::SkyboxPass(const atcg::ref_ptr<TextureCube>& skybox) : RenderPass("SkyboxPass")
{
    if(skybox) _data.setValue("skybox", skybox);
    registerOutput("framebuffer", nullptr);
    setRenderFunction(
        [](Dictionary& context, const Dictionary&, Dictionary& data, Dictionary&)
        {
            bool has_skybox = context.getValueOr<bool>("has_skybox", false);

            if(has_skybox && data.contains("skybox"))
            {
                auto _skybox = data.getValue<atcg::ref_ptr<atcg::TextureCube>>("skybox");
                atcg::Renderer::drawSkybox(_skybox, context.getValue<atcg::ref_ptr<Camera>>("camera"));
            }
        });
}
}    // namespace atcg