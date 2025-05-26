#pragma once

#include <Renderer/RenderPass.h>
#include <Renderer/Texture.h>

namespace atcg
{
class SkyboxPass : public RenderPass
{
public:
    SkyboxPass(const atcg::ref_ptr<TextureCube>& skybox = nullptr);

private:
};
}    // namespace atcg