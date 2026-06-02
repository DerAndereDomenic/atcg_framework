#pragma once

#include <Renderer/RenderPass.h>
#include <Renderer/Texture.h>

namespace atcg
{
/**
 * @brief A RenderPass that attaches the input to an output framebuffer.
 */
class OutputPass : public RenderPass
{
public:
    /**
     * @brief Constructor.
     *
     * @param properties A dictionary of properties that can be used to configure the render pass
     */
    OutputPass(Dictionary& properties);

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

    /**
     * @brief Get the output framebuffer
     *
     * @return The output framebuffer
     */
    ATCG_INLINE const atcg::ref_ptr<Framebuffer>& outputFBO() const { return _output_fbo; }

    /**
     * @brief Set the output framebuffer
     *
     * @param fbo The output framebuffer
     */
    ATCG_INLINE void setOutputFBO(const atcg::ref_ptr<Framebuffer>& fbo) { _output_fbo = fbo; }

private:
    atcg::ref_ptr<Framebuffer> _output_fbo;
};
}    // namespace atcg