#include <Renderer/RenderPasses/SkyboxPass.h>

#include <Renderer/Renderer.h>

namespace atcg
{
SkyboxPass::SkyboxPass(const atcg::ref_ptr<Skybox>& skybox) : RenderPass(RenderTargetDesc(), "SkyboxPass")
{
    initRenderPass(skybox);
}

SkyboxPass::SkyboxPass(const RenderTargetDesc& desc, const atcg::ref_ptr<Skybox>& skybox)
    : RenderPass(desc, "SkyboxPass")
{
    initRenderPass(skybox);
}

void SkyboxPass::initRenderPass(const atcg::ref_ptr<Skybox>& skybox)
{
    _data.setValue("skybox", skybox);
    registerOutput("skybox", skybox);
    registerOutput("framebuffer", atcg::make_ref<atcg::ref_ptr<Framebuffer>>(nullptr));

    setSetupFunction(
        [this](Dictionary& context, Dictionary& data, Dictionary& output_data)
        {
            auto renderer =
                context.getValueOr("renderer", atcg::SystemRegistry::instance()->getSystem<RendererSystem>());

            if(_render_target.mode == RenderTargetMode::RENDER_TARGET_OWN_FRAMEBUFFER)
            {
                auto framebuffer = output_data.getValue<atcg::ref_ptr<atcg::ref_ptr<Framebuffer>>>("framebuffer");

                *framebuffer = Framebuffer::create(_render_target.target_spec);    // TODO
            }
        });

    setRenderFunction(
        [this](Dictionary& context, const Dictionary& inputs, Dictionary& data, Dictionary& outputs)
        {
            auto renderer =
                context.getValueOr("renderer", atcg::SystemRegistry::instance()->getSystem<RendererSystem>());
            bool has_skybox = context.getValueOr<bool>("has_skybox", false);
            auto _skybox    = data.getValue<atcg::ref_ptr<atcg::Skybox>>("skybox");

            if(has_skybox && _skybox)
            {
                if(_render_target.mode != RenderTargetMode::RENDER_TARGET_BOUND_FRAMEBUFFER)
                {
                    const Dictionary& dict =
                        _render_target.mode == RenderTargetMode::RENDER_TARGET_INPUT_FRAMEBUFFER ? inputs : outputs;
                    auto framebuffer = *dict.getValue<atcg::ref_ptr<atcg::ref_ptr<Framebuffer>>>("framebuffer");
                    framebuffer->use();
                }
                renderer->drawSkybox(_skybox->getSkyboxCubeMap(), context.getValue<atcg::ref_ptr<Camera>>("camera"));
            }
        });
}
}    // namespace atcg