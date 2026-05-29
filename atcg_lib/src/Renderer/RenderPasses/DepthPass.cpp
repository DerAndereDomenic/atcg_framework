#include <Renderer/RenderPasses/DepthPass.h>

#include <Renderer/Renderer.h>
#include <Scene/Components.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{
DepthPass::DepthPass(const CullMode cull_mode) : RenderPass("DepthPass"), _cull_mode(cull_mode) {}

RenderPassReflection DepthPass::reflect(const CompileData& ctx)
{
    RenderPassReflection reflection;

    ResourceDescription desc;
    desc.type           = ResourceType::Texture;
    desc.texture.type   = TextureType::TEXTURE_2D;
    desc.texture.format = TextureFormat::DEPTH;

    uint32_t depth_handle = reflection.addOutput("depth_buffer", desc);

    reflection.setOutputFramebufferData(TextureSizeHint::FULL_FRAMEBUFFER,
                                        TextureSizeHint::FULL_FRAMEBUFFER,
                                        {depth_handle});

    return reflection;
}

void DepthPass::execute(const RenderContext& ctx, const ResourceTable& resources)
{
    auto _renderer = ctx.renderer;
    auto scene     = ctx.scene;
    auto camera    = ctx.camera;
    const auto& view =
        scene->getAllEntitiesWith<atcg::TransformComponent, atcg::GeometryComponent, atcg::MeshRenderComponent>();

    auto target = resources.getTargetFBO();

    GraphicsCommand::beginRenderPass(target);
    GraphicsCommand::clear();

    auto depth_pass_shader = _renderer->getShaderManager()->getShader("depth_pass_simple");

    GraphicsPipeline pipeline = GraphicsPipeline()
                                    .setShader(depth_pass_shader)
                                    .setRasterizerState(RasterizerState().enableCulling(true).setCullMode(_cull_mode));

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
}

}    // namespace atcg