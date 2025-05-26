#pragma once

#include <Renderer/RenderPass.h>
#include <Renderer/Texture.h>
#include <DataStructure/Skybox.h>

namespace atcg
{
class SkyboxPass : public RenderPass
{
public:
    SkyboxPass(const atcg::ref_ptr<Skybox>& skybox = nullptr);

private:
};
}    // namespace atcg