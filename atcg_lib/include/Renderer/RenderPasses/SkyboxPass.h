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
 *
 * data:
 * * data<atcg::ref_ptr<Skybox>>["skybox"] - The skybox data (set by constructor)
 *
 * inputs:
 * * None
 *
 * outputs:
 * * outputs<atcg::ref_ptr<Skybox>>["skybox"] - The skybox used. Might be a dummy skybox
 *
 */
class SkyboxPass : public RenderPass
{
public:
    /**
     * @brief Constructor
     *
     * @param skybox The skybox that should be rendered
     */
    SkyboxPass(const atcg::ref_ptr<Skybox>& skybox = nullptr);

private:
};
}    // namespace atcg