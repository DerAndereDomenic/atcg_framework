#pragma once

#include <Renderer/RenderPass.h>
#include <Renderer/Texture.h>
#include <DataStructure/Skybox.h>
#include <Scene/ComponentRenderer.h>

namespace atcg
{

/**
 * @brief A RenderPass that applys tonemapping.
 *
 * This pass reads the following variables from the context:
 * * context<RendererSystem*>["renderer"] - The renderer
 *
 * data:
 * * data<atcg::ref_ptr<ref_ptr<Framebuffer>>>["target"] - The target if RenderTargetMode is set to
 * RENDER_TARGET_OWN_FRAMEBUFFER
 *
 * inputs:
 * * inputs<atcg::ref_ptr<ref_ptr<Framebuffer>>>["framebuffer"] - The target if RenderTargetMode is set to
 * RENDER_TARGET_INPUTFRAMEBUFFER
 * * inputs<atcg::ref_ptr<ref_ptr<Framebuffer>>>["hdr_buffer"] - The HDR buffer to apply tonemapping on
 *
 * outputs:
 * * outputs<ref_ptr<ref_ptr<Framebuffer>>>["framebuffer"] - The target framebuffer
 *
 */
class TonemapPass : public RenderPass
{
public:
    /**
     * @brief Constructor.
     *
     * @param desc The Render target description
     */
    TonemapPass(const RenderTargetDesc& desc = {});

private:
    void initRenderPass();
};
}    // namespace atcg