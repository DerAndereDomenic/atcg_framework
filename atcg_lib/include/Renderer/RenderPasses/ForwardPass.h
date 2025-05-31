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
 * * context<Scene*>["scene"] - The scene
 * * context<ref_ptr<Camera>>["camera"] - The camera
 * * context<bool>["has_skybox"] - If a skybox should be used for ibl (default: false)
 *
 * data:
 * * data<atcg::ref_ptr<ComponentRenderer>>["component_renderer"] - The component renderer to render the components.
 * * data<atcg::ref_ptr<Skybox>>["skybox"] - The skybox data. This is a valid skybox even if nullptr is passed to the
 * constructor. Therefore, if "use_ibl" is set to true but no skybox is given, the Renderer will perform ibl with an
 * empty skybox.
 *
 * inputs:
 * * inputs<ref_ptr<ref_ptr<TextureCubeArray>>>["point_light_depth_maps"] - A cube map array with one cube map per
 * light source. This is a double pointer because depending on the (dynamic) number of light sources, this has to be
 * recreated on the fly. If this is not present, no shadow mapping will be performed.
 *
 * outputs:
 * * outputs["framebuffer"] - nullptr - This RenderPass renders to the currently bound framebuffer. The "framebuffer"
 * port is only used to connect to other render passes.
 */
class ForwardPass : public RenderPass
{
public:
    /**
     * @brief Constructor.
     * If a skybox is given, the data is used for ibl
     *
     * @param skybox The skybox
     */
    ForwardPass(const atcg::ref_ptr<Skybox>& skybox = nullptr);

private:
    atcg::ref_ptr<atcg::Skybox> _skybox;
};
}    // namespace atcg