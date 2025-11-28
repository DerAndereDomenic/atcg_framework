#include <Renderer/RenderPasses/BlitPass.h>

#include <Renderer/Renderer.h>
#include <Scene/Components.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{

BlitPass::BlitPass(const RenderTargetDesc& desc) : RenderPass(desc, "BlitPass")
{
    initRenderPass();
}

void BlitPass::initRenderPass()
{
    registerOutput("framebuffer", atcg::make_ref<atcg::ref_ptr<Framebuffer>>(nullptr));
    setSetupFunction(
        [this](Dictionary& context, Dictionary& data, Dictionary& output)
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

            auto input  = *inputs.getValue<atcg::ref_ptr<atcg::ref_ptr<Framebuffer>>>("framebuffer");
            auto target = prepareFramebuffer(context, inputs, data, outputs);

            auto output_framebuffer = outputs.getValue<atcg::ref_ptr<atcg::ref_ptr<Framebuffer>>>("framebuffer");
            *output_framebuffer     = target;

            target->use();
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
            target->blit(input);
        });
}
}    // namespace atcg