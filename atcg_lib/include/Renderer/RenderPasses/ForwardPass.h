#pragma once

#include <Renderer/RenderPass.h>
#include <Renderer/Texture.h>

namespace atcg
{
class ForwardPass : public RenderPass
{
public:
    ForwardPass(const atcg::ref_ptr<TextureCube>& irradiance_cubemap  = nullptr,
                const atcg::ref_ptr<TextureCube>& prefiltered_cubemap = nullptr);

private:
};
}    // namespace atcg