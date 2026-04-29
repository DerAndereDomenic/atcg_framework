#include <Renderer/RenderPasses/DepthPass.h>

#include <Renderer/Renderer.h>
#include <Scene/Components.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{
DepthPass::DepthPass(const CullMode cull_mode, const RenderTargetDesc& desc) : RenderPass(desc, "DepthPass")
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
        [this, cull_mode](Dictionary& context, const Dictionary& inputs, Dictionary& data, Dictionary& outputs)
        {
            auto _renderer =
                context.getValueOr("renderer", atcg::SystemRegistry::instance()->getSystem<RendererSystem>());
            auto scene       = context.getValue<atcg::ref_ptr<Scene>>("scene");
            auto camera      = context.getValue<atcg::ref_ptr<Camera>>("camera");
            const auto& view = scene->getAllEntitiesWith<atcg::TransformComponent,
                                                         atcg::GeometryComponent,
                                                         atcg::MeshRenderComponent>();


            Dictionary auxiliary;

            auto output_framebuffer = outputs.getValue<atcg::ref_ptr<atcg::ref_ptr<Framebuffer>>>("framebuffer");
            auto target             = prepareFramebuffer(context, inputs, data, outputs);
            *output_framebuffer     = target;

            GraphicsCommand::beginRenderPass(target);
            if(_render_target.clear)
            {
                GraphicsCommand::clear();
                //  We assume that this is an entity buffer, better solution?
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

            auto depth_pass_shader = _renderer->getShaderManager()->getShader("depth_pass_simple");

            GraphicsPipeline pipeline =
                GraphicsPipeline()
                    .setShader(depth_pass_shader)
                    .setRasterizerState(RasterizerState().enableCulling(true).setCullMode(cull_mode));

            for(auto e: view)
            {
                atcg::Entity entity(e, scene.get());

                auto& renderer = entity.getComponent<MeshRenderComponent>();
                if(!renderer.visible)
                {
                    continue;
                }

                auto& geometry = entity.getComponent<atcg::GeometryComponent>();
                if(!geometry.graph())
                {
                    continue;
                }

                _renderer->drawVAO(geometry.graph()->getVerticesArray(),
                                   camera,
                                   entity.getComponent<atcg::TransformComponent>().getModel(),
                                   pipeline,
                                   geometry.graph()->n_vertices());
            }

            GraphicsCommand::endRenderPass();
        });
}
}    // namespace atcg