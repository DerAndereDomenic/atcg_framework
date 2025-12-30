#include <Renderer/RenderPasses/SkyboxPass.h>

#include <Renderer/Renderer.h>

namespace atcg
{
SkyboxPass::SkyboxPass() : RenderPass(RenderTargetDesc(), "SkyboxPass")
{
    initRenderPass();
}

SkyboxPass::SkyboxPass(const RenderTargetDesc& desc) : RenderPass(desc, "SkyboxPass")
{
    initRenderPass();
}

void SkyboxPass::initRenderPass()
{
    registerOutput("framebuffer", atcg::make_ref<atcg::ref_ptr<Framebuffer>>(nullptr));
    registerOutput("skybox", atcg::make_ref<atcg::ref_ptr<Skybox>>(nullptr));

    setSetupFunction(
        [this](Dictionary& context, Dictionary& data, Dictionary& output_data)
        {
            if(_render_target.mode == RenderTargetMode::RENDER_TARGET_OWN_FRAMEBUFFER)
            {
                data.setValue("target", atcg::make_ref<atcg::ref_ptr<Framebuffer>>(nullptr));
            }
        });

    setRenderFunction(
        [this](Dictionary& context, const Dictionary& inputs, Dictionary& data, Dictionary& outputs)
        {
            auto renderer =
                context.getValueOr("renderer", atcg::SystemRegistry::instance()->getSystem<RendererSystem>());
            bool has_skybox = context.getValueOr<bool>("has_skybox", false);
            auto _skybox    = context.getValueOr<atcg::ref_ptr<atcg::Skybox>>("skybox", nullptr);

            auto output_framebuffer = outputs.getValue<atcg::ref_ptr<atcg::ref_ptr<Framebuffer>>>("framebuffe"
                                                                                                  "r");
            auto output_skybox      = outputs.getValue<atcg::ref_ptr<atcg::ref_ptr<Skybox>>>("skybox");
            auto target             = prepareFramebuffer(context, inputs, data, outputs);
            *output_framebuffer     = target;
            *output_skybox          = _skybox;
            if(_render_target.clear)
            {
                renderer->clear();

                // We assume that this is an entity buffer, better solution?
                if(target->numColorAttachements() > 1 &&
                   target->getColorAttachement(1)->getSpecification().format == TextureFormat::RINT)
                {
                    int value = -1;
                    target->getColorAttachement(1)->fill(&value);
                }

                if(target->numColorAttachements() > 2 &&
                   target->getColorAttachement(2)->getSpecification().format == TextureFormat::RINT8)
                {
                    uint8_t value = 0;
                    target->getColorAttachement(2)->fill(&value);
                }
            }

            renderer->beginRenderPass(target);
            if(has_skybox && _skybox)
            {
                renderer->drawSkybox(_skybox->getSkyboxCubeMap(), context.getValue<atcg::ref_ptr<Camera>>("camera"));
            }
            renderer->endRenderPass();
        });
}
}    // namespace atcg