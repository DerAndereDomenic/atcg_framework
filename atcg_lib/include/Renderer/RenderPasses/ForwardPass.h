#pragma once

#include <Renderer/RenderPass.h>
#include <Renderer/Texture.h>
#include <DataStructure/Skybox.h>
#include <Scene/ComponentRenderer.h>

namespace atcg
{
class ForwardPass : public RenderPass
{
public:
    ForwardPass(const atcg::ref_ptr<Skybox>& skybox = nullptr);

private:
    atcg::ref_ptr<atcg::Skybox> _skybox;
};
}    // namespace atcg