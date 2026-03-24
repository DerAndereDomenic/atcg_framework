#pragma once

#include <Renderer/RenderPass.h>

namespace atcg
{

class VolumePass : public RenderPass
{
public:
    VolumePass(const RenderTargetDesc& desc = {});
};

}