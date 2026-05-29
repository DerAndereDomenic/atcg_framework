#include <Renderer/RenderPasses/BlitPass.h>

#include <Renderer/Renderer.h>
#include <Scene/Components.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{

BlitPass::BlitPass() : RenderPass("BlitPass")
{
    std::vector<atcg::Vertex> vertices = {atcg::Vertex(glm::vec3(-1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, 1, 0)),
                                          atcg::Vertex(glm::vec3(-1, 1, 0))};

    std::vector<glm::u32vec3> edges = {glm::u32vec3(0, 1, 2), glm::u32vec3(0, 2, 3)};

    _screen_quad = atcg::Graph::createTriangleMesh(vertices, edges);
}

RenderPassReflection BlitPass::reflect(const CompileData& ctx)
{
    RenderPassReflection reflection;


    reflection.addInput("input_color_buffer");
    reflection.addInput("input_depth_buffer");
    reflection.addInput("input_entity_buffer");
    reflection.addInput("input_stencil_buffer");


    ResourceDescription desc_color;
    desc_color.type           = ResourceType::Texture;
    desc_color.texture.type   = TextureType::TEXTURE_2D;
    desc_color.texture.format = TextureFormat::RGBAFLOAT;

    uint32_t color_handle = reflection.addOutput("out_color_buffer", desc_color);

    ResourceDescription desc_depth;
    desc_depth.type           = ResourceType::Texture;
    desc_depth.texture.type   = TextureType::TEXTURE_2D;
    desc_depth.texture.format = TextureFormat::DEPTH;

    uint32_t depth_handle = reflection.addOutput("out_depth_buffer", desc_depth);

    ResourceDescription desc_entity;
    desc_entity.type           = ResourceType::Texture;
    desc_entity.texture.type   = TextureType::TEXTURE_2D;
    desc_entity.texture.format = TextureFormat::RINT;

    uint32_t entity_handle = reflection.addOutput("out_entity_buffer", desc_entity);

    ResourceDescription desc_stencil;
    desc_stencil.type           = ResourceType::Texture;
    desc_stencil.texture.type   = TextureType::TEXTURE_2D;
    desc_stencil.texture.format = TextureFormat::RINT8;

    uint32_t stencil_handle = reflection.addOutput("out_stencil_buffer", desc_stencil);

    reflection.setOutputFramebufferData(TextureSizeHint::FULL_FRAMEBUFFER,
                                        TextureSizeHint::FULL_FRAMEBUFFER,
                                        {color_handle, entity_handle, stencil_handle, depth_handle});

    return reflection;
}

void BlitPass::execute(const RenderContext& ctx, const ResourceTable& resources)
{
    auto renderer = ctx.renderer;

    auto input_color   = resources.getTexture<Texture2DMultiSample>("input_color_buffer");
    auto input_depth   = resources.getTexture<Texture2DMultiSample>("input_depth_buffer");
    auto input_entity  = resources.getTexture<Texture2DMultiSample>("input_entity_buffer");
    auto input_stencil = resources.getTexture<Texture2DMultiSample>("input_stencil_buffer");

    auto target = resources.getTargetFBO();

    GraphicsCommand::beginRenderPass(target);

    GraphicsCommand::clear();

    {
        int value = -1;
        target->getColorAttachement(1)->fill(&value);
    }

    {
        uint8_t value = 0;
        target->getColorAttachement(2)->fill(&value);
    }

    auto shader = renderer->getShaderManager()->getShader("blit");

    GraphicsPipeline pipeline = GraphicsPipeline()
                                    .setShader(shader)
                                    .setPrimitiveTopology(PrimitiveTopology::ATCG_TRIANGLES)
                                    .setRasterizerState(RasterizerState().setDepthState(
                                        DepthState().setDepthFunction(DepthFunction::ATCG_ALWAYS)));

    uint32_t color_id   = renderer->popTextureID();
    uint32_t depth_id   = renderer->popTextureID();
    uint32_t entity_id  = renderer->popTextureID();
    uint32_t stencil_id = renderer->popTextureID();

    shader->setInt("in_color", color_id);
    shader->setInt("in_depth", depth_id);
    shader->setInt("in_entity", entity_id);
    shader->setInt("in_stencil", stencil_id);

    GraphicsCommand::bindTexture(color_id, input_color);
    GraphicsCommand::bindTexture(depth_id, input_depth);
    GraphicsCommand::bindTexture(entity_id, input_entity);
    GraphicsCommand::bindTexture(stencil_id, input_stencil);

    renderer->drawVAO(_screen_quad->getVerticesArray(), {}, glm::mat4(1), pipeline, _screen_quad->n_vertices());

    renderer->pushTextureID(color_id);
    renderer->pushTextureID(depth_id);
    renderer->pushTextureID(entity_id);
    renderer->pushTextureID(stencil_id);

    GraphicsCommand::endRenderPass();
}
}    // namespace atcg