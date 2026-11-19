#pragma once

#include <Renderer/RenderPass.h>
#include <Renderer/Shader.h>
#include <Renderer/Texture.h>
#include <Scene/ComponentRenderer.h>
#include <Renderer/RenderPassRegistry.h>

namespace atcg
{
/**
 * @brief A RenderPass that renders only the currently selected entities into an entity id mask.
 *
 * The mask is a single RINT texture ("selection_mask") that holds the id of a selected entity where it covers the
 * pixel and -1 everywhere else. No depth test is performed, so the mask contains the full silhouettes of the selected
 * entities, even where they are occluded. The active entity is drawn last, so its id wins where selected entities
 * overlap.
 */
class ATCG_API SelectionMaskPass : public RenderPass
{
public:
    /**
     * @brief Constructor.
     *
     * @param properties A dictionary of properties that can be used to configure the render pass
     */
    SelectionMaskPass(Dictionary& properties);

    /**
     * @brief Reflect the render pass. This function describes the inputs, outputs and framebuffer data of this render
     * pass. This is used by the render graph to generate the resource tables and framebuffers for this render pass.
     *
     * @param ctx The compile data
     * @return The reflection data of this render pass
     */
    virtual RenderPassReflection reflect(const CompileData& ctx) override;

    /**
     * @brief Execute the render pass. This function is called by the render graph to execute this render pass. The
     * resources used by this render pass are passed in the resource table.
     *
     * @param ctx The render context holding per-frame data
     * @param resources The resource table holding the resources for this render pass
     */
    virtual void execute(const RenderContext& ctx, const ResourceTable& resources) override;

    static void registerRenderPass(RenderPassRegistry::Registry* registry);

private:
    template<typename ComponentT>
    void renderWithMaskShader(const RenderContext& ctx,
                              Entity entity,
                              const atcg::ref_ptr<Shader>& mask_shader,
                              Dictionary& auxiliary);

    void renderEntityMask(const RenderContext& ctx, Entity entity, Dictionary& auxiliary);

    atcg::ref_ptr<Shader> _mask_shader_base;
    atcg::ref_ptr<Shader> _mask_shader_instanced;
    atcg::ref_ptr<Shader> _mask_shader_cylinder;
    atcg::ref_ptr<Shader> _mask_shader_edge;
    atcg::ref_ptr<Shader> _mask_shader_circle;
};
}    // namespace atcg
