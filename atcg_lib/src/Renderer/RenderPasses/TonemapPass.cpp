#include <Renderer/RenderPasses/TonemapPass.h>

#include <Renderer/Renderer.h>
#include <Scene/Components.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{

TonemapPass::TonemapPass() : RenderPass("TonemapPass")
{
    std::vector<atcg::Vertex> vertices = {atcg::Vertex(glm::vec3(-1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, 1, 0)),
                                          atcg::Vertex(glm::vec3(-1, 1, 0))};

    std::vector<glm::u32vec3> edges = {glm::u32vec3(0, 1, 2), glm::u32vec3(0, 2, 3)};

    _screen_quad = atcg::Graph::createTriangleMesh(vertices, edges);
}

RenderPassReflection TonemapPass::reflect(const CompileData& ctx)
{
    RenderPassReflection reflection;

    reflection.addInput("hdr");

    ResourceDescription out_desc;
    out_desc.type           = ResourceType::Texture;
    out_desc.texture.type   = TextureType::TEXTURE_2D;
    out_desc.texture.format = TextureFormat::RGBA;
    uint32_t output_handle  = reflection.addOutput("output_color", out_desc);

    reflection.addInput("in_stencil_buffer");

    reflection.setOutputFramebufferData(TextureSizeHint::FULL_FRAMEBUFFER,
                                        TextureSizeHint::FULL_FRAMEBUFFER,
                                        {output_handle});

    return reflection;
}

void TonemapPass::execute(const RenderContext& ctx, const ResourceTable& resources)
{
    auto renderer = ctx.renderer;

    auto scene      = ctx.scene;
    auto hdr        = resources.getTexture<Texture2D>("hdr");
    auto in_stencil = resources.getTexture<Texture2D>("in_stencil_buffer");

    auto target = resources.getTargetFBO();

    GraphicsCommand::beginRenderPass(target);
    GraphicsCommand::clear();
    // target->blit(hdr, false, true);    // Copy depth

    auto shader = renderer->getShaderManager()->getShader("tonemap");

    GraphicsPipeline pipeline =
        GraphicsPipeline()
            .setShader(shader)
            .setPrimitiveTopology(PrimitiveTopology::ATCG_TRIANGLES)
            .setRasterizerState(RasterizerState().setDepthState(DepthState().enableDepthTesting(false)));

    uint32_t screen_id  = renderer->popTextureID();
    uint32_t stencil_id = renderer->popTextureID();

    shader->setInt("screen_texture", screen_id);
    shader->setInt("stencil_texture", stencil_id);
    shader->setFloat("exposure", scene->getCamera() ? scene->getCamera()->getIntrinsics().getExposure() : 1.0f);

    GraphicsCommand::bindTexture(screen_id, hdr);
    GraphicsCommand::bindTexture(stencil_id, in_stencil);

    renderer->drawVAO(_screen_quad->getVerticesArray(), {}, glm::mat4(1), pipeline, _screen_quad->n_vertices());

    renderer->pushTextureID(screen_id);
    renderer->pushTextureID(stencil_id);

    GraphicsCommand::endRenderPass();
}

}    // namespace atcg