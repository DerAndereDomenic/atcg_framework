#pragma once

#include <Renderer/RenderPass.h>

namespace atcg
{
/**
 * @brief A RenderPass that creates and renders shadow maps for all PointLightComponents in a scene
 *
 * This pass reads the following variables from the context:
 * * context<RendererSystem*>["renderer"] - The renderer
 * * context<Scene*>["scene"] - The scene
 *
 * data:
 * * data<atcg::ref_ptr<Framebuffer>>["point_light_framebuffer"] - The internal framebuffer used for the depth passes
 *
 * inputs:
 * * None
 *
 * outputs:
 * * outputs<ref_ptr<ref_ptr<TextureCubeArray>>>["point_light_depth_maps"] - A cube map array with one cube map per
 * light source. This is a double pointer because depending on the (dynamic) number of light sources, this has to be
 * recreated on the fly.
 *
 */
class ShadowPass : public RenderPass
{
public:
    /**
     * @brief Constructor
     */
    ShadowPass();

private:
};
}    // namespace atcg