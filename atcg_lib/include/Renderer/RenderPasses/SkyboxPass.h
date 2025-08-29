#pragma once

#include <Renderer/RenderPass.h>
#include <Renderer/Texture.h>
#include <DataStructure/Skybox.h>

namespace atcg
{
/**
 * @brief A RenderPass that renders a skybox.
 *
 * This pass reads the following variables from the context:
 * * context<RendererSystem*>["renderer"] - The renderer
 * * context<bool>["has_skybox"] - If the skybox is present
 * * context<atcg::ref_ptr<Skybox>>["skybox"] - The skybox used
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
 * * outputs<ref_ptr<ref_ptr<Skybox>>>["skybox"] - The skybox used. Might be a dummy skybox
 * * outputs<ref_ptr<ref_ptr<Framebuffer>>>["framebuffer"] - The target framebuffer
 *
 */
class SkyboxPass : public RenderPass
{
public:
    /**
     * @brief Constructor
     */
    SkyboxPass();

    /**
     * @brief Constructor
     *
     * @param desc The render target description
     */
    SkyboxPass(const RenderTargetDesc& desc);

private:
    void initRenderPass();
};
}    // namespace atcg