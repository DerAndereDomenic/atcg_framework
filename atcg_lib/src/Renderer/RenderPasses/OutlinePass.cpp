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
    reflection.addInput("input_entity_buffer");


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

    auto input_color  = resources.getTexture<Texture2D>("input_color_buffer");
    auto input_entity = resources.getTexture<Texture2D>("input_entity_buffer");

    auto target = resources.getTargetFBO();

    GraphicsCommand::beginRenderPass(target);

    GraphicsCommand::clear();

    auto shader = renderer->getShaderManager()->getShader("outline");

    GraphicsPipeline pipeline =
        GraphicsPipeline()
            .setShader(shader)
            .setPrimitiveTopology(PrimitiveTopology::ATCG_TRIANGLES)
            .setRasterizerState(RasterizerState().setDepthState(DepthState().enableDepthWrite(false)));

    uint32_t color_id  = renderer->popTextureID();
    uint32_t entity_id = renderer->popTextureID();

    shader->setInt("input_color_buffer", color_id);
    shader->setInt("entity_ids", entity_id);
    shader->setInt("selected_entity_id", ctx.scene->getSelectedEntity().entity_handle());

    GraphicsCommand::bindTexture(color_id, input_color);
    GraphicsCommand::bindTexture(entity_id, input_entity);

    renderer->drawVAO(_screen_quad->getVerticesArray(), {}, glm::mat4(1), pipeline, _screen_quad->n_vertices());

    renderer->pushTextureID(color_id);
    renderer->pushTextureID(entity_id);

    GraphicsCommand::endRenderPass();
}
}    // namespace atcg