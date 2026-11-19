#include <Renderer/RenderPasses/OutlinePass.h>

#include <Renderer/Renderer.h>
#include <Scene/Components.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{

OutlinePass::OutlinePass(Dictionary& properties) : RenderPass(properties, "OutlinePass")
{
    std::vector<atcg::Vertex> vertices = {atcg::Vertex(glm::vec3(-1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, 1, 0)),
                                          atcg::Vertex(glm::vec3(-1, 1, 0))};

    std::vector<glm::u32vec3> edges = {glm::u32vec3(0, 1, 2), glm::u32vec3(0, 2, 3)};

    _screen_quad = atcg::Graph::createTriangleMesh(vertices, edges);
}

RenderPassReflection OutlinePass::reflect(const CompileData& ctx)
{
    RenderPassReflection reflection;


    reflection.addInput("input_color_buffer");
    reflection.addInput("input_selection_mask");


    ResourceDescription desc_color;
    desc_color.type           = ResourceType::Texture;
    desc_color.texture.type   = TextureType::TEXTURE_2D;
    desc_color.texture.format = TextureFormat::RGBA;

    uint32_t color_handle = reflection.addOutput("out_color_buffer", desc_color);

    reflection.setOutputFramebufferData(TextureSizeHint::FULL_FRAMEBUFFER,
                                        TextureSizeHint::FULL_FRAMEBUFFER,
                                        {color_handle});

    return reflection;
}

void OutlinePass::execute(const RenderContext& ctx, const ResourceTable& resources)
{
    auto renderer = ctx.renderer;

    auto input_color    = resources.getTexture<Texture2D>("input_color_buffer");
    auto selection_mask = resources.getTexture<Texture2D>("input_selection_mask");

    auto target = resources.getTargetFBO();

    GraphicsCommand::beginRenderPass(target);

    GraphicsCommand::clear();

    auto shader = renderer->getShaderManager()->getShader("outline");

    GraphicsPipeline pipeline =
        GraphicsPipeline()
            .setShader(shader)
            .setPrimitiveTopology(PrimitiveTopology::ATCG_TRIANGLES)
            .setRasterizerState(RasterizerState().setDepthState(DepthState().enableDepthWrite(false)));

    uint32_t color_id = renderer->popTextureID();
    uint32_t mask_id  = renderer->popTextureID();

    shader->setInt("input_color_buffer", color_id);
    shader->setInt("selection_mask", mask_id);
    shader->setInt("active_entity_id", ctx.scene->getActiveEntity().entity_handle());

    GraphicsCommand::bindTexture(color_id, input_color);
    GraphicsCommand::bindTexture(mask_id, selection_mask);

    renderer->drawVAO(_screen_quad->getVerticesArray(), {}, glm::mat4(1), pipeline, _screen_quad->n_vertices());

    renderer->pushTextureID(color_id);
    renderer->pushTextureID(mask_id);

    GraphicsCommand::endRenderPass();
}

void OutlinePass::registerRenderPass(RenderPassRegistry::Registry* registry)
{
    ATCG_REGISTER_RENDER_PASS(registry, "OutlinePass", OutlinePass);
}


}    // namespace atcg