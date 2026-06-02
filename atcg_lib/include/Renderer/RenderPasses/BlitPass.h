#pragma once

#include <Renderer/RenderPass.h>
#include <Renderer/Texture.h>
#include <DataStructure/Skybox.h>
#include <Scene/ComponentRenderer.h>

namespace atcg
{

/**
 * @brief A RenderPass that blits the input buffers to the output framebuffer. This is used as the final pass in the
 * render graph to present the rendered image to the screen.
 */
class ATCG_API BlitPass : public RenderPass
{
public:
    /**
     * @brief Constructor.
     *
     * @param properties A dictionary of properties that can be used to configure the render pass
     */
    BlitPass(Dictionary& properties);

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

private:
    atcg::ref_ptr<Graph> _screen_quad;
};
}    // namespace atcg