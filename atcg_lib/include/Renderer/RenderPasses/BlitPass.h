#pragma once

#include <Renderer/RenderPass.h>
#include <Renderer/Texture.h>
#include <DataStructure/Skybox.h>
#include <Scene/ComponentRenderer.h>

namespace atcg
{

/**
 * @brief A RenderPass that blits two framebuffer.
 *
 * This pass reads the following variables from the context:
 * * context<RendererSystem*>["renderer"] - The renderer
 *
 * data:
 * * data<atcg::ref_ptr<ref_ptr<Framebuffer>>>["target"] - The target if RenderTargetMode is set to
 * RENDER_TARGET_OWN_FRAMEBUFFER
 *
 * inputs:
 * * * data<atcg::ref_ptr<ref_ptr<Framebuffer>>>["framebuffer"] - The target if RenderTargetMode is set to
 * RENDER_TARGET_INPUTFRAMEBUFFER
 *
 * outputs:
 * * outputs<ref_ptr<ref_ptr<Framebuffer>>>["framebuffer"] - The target framebuffer
 *
 */
class BlitPass : public RenderPass
{
public:
    /**
     * @brief Constructor.
     *
     * @param desc The Render target description
     */
    BlitPass(const RenderTargetDesc& desc = {});

private:
    void initRenderPass();
};
}    // namespace atcg