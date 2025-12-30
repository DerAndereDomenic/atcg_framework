#include <Renderer/RenderPasses/TonemapPass.h>

#include <Renderer/Renderer.h>
#include <Scene/Components.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{

TonemapPass::TonemapPass(const RenderTargetDesc& desc) : RenderPass(desc, "TonemapPass")
{
    initRenderPass();
}

void TonemapPass::initRenderPass()
{
    registerOutput("framebuffer", atcg::make_ref<atcg::ref_ptr<Framebuffer>>(nullptr));
    setSetupFunction(
        [this](Dictionary& context, Dictionary& data, Dictionary& output)
        {
            if(_render_target.mode == RenderTargetMode::RENDER_TARGET_OWN_FRAMEBUFFER)
            {
                data.setValue("target", atcg::make_ref<atcg::ref_ptr<Framebuffer>>(nullptr));
            }

            atcg::ref_ptr<Graph> quad;
            {
                std::vector<atcg::Vertex> vertices = {atcg::Vertex(glm::vec3(-1, -1, 0)),
                                                      atcg::Vertex(glm::vec3(1, -1, 0)),
                                                      atcg::Vertex(glm::vec3(1, 1, 0)),
                                                      atcg::Vertex(glm::vec3(-1, 1, 0))};

                std::vector<glm::u32vec3> edges = {glm::u32vec3(0, 1, 2), glm::u32vec3(0, 2, 3)};

                quad = atcg::Graph::createTriangleMesh(vertices, edges);
            }

            data.setValue("screen_quad", quad);
        });


    setRenderFunction(
        [this](Dictionary& context, const Dictionary& inputs, Dictionary& data, Dictionary& outputs)
        {
            auto renderer =
                context.getValueOr("renderer", atcg::SystemRegistry::instance()->getSystem<RendererSystem>());

            auto hdr    = *inputs.getValue<atcg::ref_ptr<atcg::ref_ptr<Framebuffer>>>("hdr");
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
            renderer->beginRenderPass(target);

            target->blit(hdr, false, true);    // Copy depth

            auto shader = renderer->getShaderManager()->getShader("tonemap");

            GraphicsPipeline pipeline =
                GraphicsPipeline()
                    .setShader(shader)
                    .setPrimitiveTopology(PrimitiveTopology::ATCG_TRIANGLES)
                    .setRasterizerState(RasterizerState().setDepthState(DepthState().enableDepthTesting(false)));

            uint32_t screen_id  = renderer->popTextureID();
            uint32_t entity_id  = renderer->popTextureID();
            uint32_t stencil_id = renderer->popTextureID();

            shader->setInt("screen_texture", screen_id);
            shader->setInt("entity_texture", entity_id);
            shader->setInt("stencil_texture", stencil_id);

            hdr->getColorAttachement(0)->use(screen_id);
            hdr->getColorAttachement(1)->use(entity_id);
            hdr->getColorAttachement(2)->use(stencil_id);

            auto screen_quad = data.getValue<atcg::ref_ptr<Graph>>("screen_quad");
            renderer->drawVAO(screen_quad->getVerticesArray(), {}, glm::mat4(1), pipeline, screen_quad->n_vertices());

            renderer->pushTextureID(screen_id);
            renderer->pushTextureID(entity_id);
            renderer->pushTextureID(stencil_id);

            renderer->endRenderPass();
        });
}
}    // namespace atcg