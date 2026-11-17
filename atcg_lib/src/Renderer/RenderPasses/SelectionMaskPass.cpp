#include <Renderer/RenderPasses/SelectionMaskPass.h>

#include <Core/Path.h>
#include <Renderer/Renderer.h>
#include <Scene/Components.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{

SelectionMaskPass::SelectionMaskPass(Dictionary& properties) : RenderPass(properties, "SelectionMaskPass")
{
    // Reuse the vertex (and geometry) stages of the regular component shaders so that geometry is placed exactly as in
    // the forward pass, but replace the fragment stage with a shader that only writes the entity id (no lighting).
    const std::filesystem::path shader_dir = atcg::shader_directory();
    const std::string mask_fragment        = (shader_dir / "selection_mask.fs").string();

    _mask_shader_base      = atcg::make_ref<Shader>((shader_dir / "base.vs").string(), mask_fragment);
    _mask_shader_instanced = atcg::make_ref<Shader>((shader_dir / "instanced.vs").string(), mask_fragment);
    _mask_shader_cylinder  = atcg::make_ref<Shader>((shader_dir / "cylinder_edge.vs").string(), mask_fragment);
    _mask_shader_edge      = atcg::make_ref<Shader>((shader_dir / "edge.vs").string(),
                                               mask_fragment,
                                               (shader_dir / "edge.gs").string());
}

RenderPassReflection SelectionMaskPass::reflect(const CompileData& ctx)
{
    RenderPassReflection reflection;

    // The mask is used for a screen space effect only, so it is never multisampled
    ResourceDescription desc_mask;
    desc_mask.type                        = ResourceType::Texture;
    desc_mask.texture.type                = TextureType::TEXTURE_2D;
    desc_mask.texture.format              = TextureFormat::RINT;
    desc_mask.texture.sampler.wrap_mode   = TextureWrapMode::CLAMP_TO_EDGE;
    desc_mask.texture.sampler.filter_mode = TextureFilterMode::NEAREST;

    uint32_t mask_handle = reflection.addOutput("selection_mask", desc_mask);

    // No depth attachment: the selected entity is drawn completely, regardless of what is in front of it
    reflection.setOutputFramebufferData(TextureSizeHint::FULL_FRAMEBUFFER,
                                        TextureSizeHint::FULL_FRAMEBUFFER,
                                        {mask_handle});

    return reflection;
}

template<typename ComponentT>
void SelectionMaskPass::renderWithMaskShader(const RenderContext& ctx,
                                             Entity entity,
                                             const atcg::ref_ptr<Shader>& mask_shader,
                                             Dictionary& auxiliary)
{
    // The component renderers pick up "override_shader" and set entityID (and point_size/edge_radius) on it
    auxiliary.setValue("override_shader", mask_shader);
    renderComponent<ComponentT>(ctx.renderer, entity, ctx.camera, auxiliary);
}

void SelectionMaskPass::execute(const RenderContext& ctx, const ResourceTable& resources)
{
    auto target = resources.getTargetFBO();

    GraphicsCommand::beginRenderPass(target);

    GraphicsCommand::clear();
    {
        int value = -1;
        target->getColorAttachement(0)->fill(&value);
    }

    Entity selected = ctx.scene->getSelectedEntity();
    if(!selected)
    {
        GraphicsCommand::endRenderPass();
        return;
    }

    Dictionary auxiliary;
    auxiliary.setValue("draw_cameras", ctx.draw_cameras);
    auxiliary.setValue("has_skybox", false);

    renderWithMaskShader<MeshRenderComponent>(ctx, selected, _mask_shader_base, auxiliary);
    renderWithMaskShader<PointRenderComponent>(ctx, selected, _mask_shader_base, auxiliary);
    renderWithMaskShader<PointSphereRenderComponent>(ctx, selected, _mask_shader_base, auxiliary);
    renderWithMaskShader<MeshLightComponent>(ctx, selected, _mask_shader_base, auxiliary);
    renderWithMaskShader<InstanceRenderComponent>(ctx, selected, _mask_shader_instanced, auxiliary);
    renderWithMaskShader<EdgeCylinderRenderComponent>(ctx, selected, _mask_shader_cylinder, auxiliary);
    renderWithMaskShader<EdgeRenderComponent>(ctx, selected, _mask_shader_edge, auxiliary);

    GraphicsCommand::endRenderPass();
}

void SelectionMaskPass::registerRenderPass(RenderPassRegistry::Registry* registry)
{
    ATCG_REGISTER_RENDER_PASS(registry, "SelectionMaskPass", SelectionMaskPass);
}

}    // namespace atcg
