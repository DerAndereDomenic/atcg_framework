#pragma once

#include <Renderer/RenderPass.h>
#include <Renderer/Texture.h>
#include <DataStructure/Skybox.h>
#include <Scene/ComponentRenderer.h>

namespace atcg
{

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