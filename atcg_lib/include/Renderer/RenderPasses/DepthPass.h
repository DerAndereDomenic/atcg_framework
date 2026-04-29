#pragma once

#include <Renderer/RenderPass.h>
#include <Renderer/Texture.h>
#include <DataStructure/Skybox.h>
#include <Scene/ComponentRenderer.h>

namespace atcg
{


class DepthPass : public RenderPass
{
public:
    /**
     * @brief Constructor.
     */
    DepthPass(const CullMode cull_mode, const RenderTargetDesc& desc = {});

private:
};
}    // namespace atcg