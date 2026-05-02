#pragma once

#include <Renderer/RenderPass.h>
#include <Renderer/Texture.h>
#include <DataStructure/Skybox.h>
#include <Scene/ComponentRenderer.h>

namespace atcg
{

/**
 * @brief A RenderPass that performs a simple forward rendering step.
 *
 * This pass reads the following variables from the context:
 * * context<RendererSystem*>["renderer"] - The renderer
 * * context<atcg::ref_ptr<Scene>>["scene"] - The scene
 * * context<ref_ptr<Camera>>["camera"] - The camera
 * * context<bool>["has_skybox"] - If a skybox should be used for ibl (default: false)
 *
 * data:
 * * data<atcg::ref_ptr<ref_ptr<Framebuffer>>>["target"] - The target framebuffer (if this renderpass owns a
 * framebuffer)
 * * data<atcg::ref_ptr<Skybox>>["dummy_skybox"] - A dummy skybox to use if no skybox is passed from a previous render
 * pass.
 *
 * inputs:
 * * inputs<ref_ptr<ref_ptr<TextureCubeArray>>>["point_light_depth_maps"] - A cube map array with one cube map per
 * light source. This is a double pointer because depending on the (dynamic) number of light sources, this has to be
 * recreated on the fly. If this is not present, no shadow mapping will be performed.
 * * inputs<ref_ptr<ref_ptr<Skybox>>>["skybox"] - The optional skybox to use for lighting
 * * inputs<ref_ptr<ref_ptr<Framebuffer>>>["framebuffer"] - The target framebuffer used if the RenderTargetMode is set
 * to RENDER_TARGET_INPUT_FRAMEBUFFER
 *
 * outputs:
 * * outputs<ref_ptr<ref_ptr<Framebuffer>>>["framebuffer"] - The output framebuffer
 */
class ForwardPass : public RenderPass
{
public:
    /**
     * @brief Constructor.
     */
    ForwardPass(const RenderTargetDesc& desc = {});

private:
    struct TransparentRenderData
    {
        atcg::Entity entity;
        float distance_to_camera;
    };
    std::vector<TransparentRenderData> _transparent_entities;
};
}    // namespace atcg