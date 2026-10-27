#pragma once

#include <DataStructure/TextureSampler.h>

namespace atcg
{
struct PerfectDielectricData
{
    TextureSampler<float> ior_texture;

    TextureWriter<float> ior_grad;

    bool optimize_ior = false;
};
}    // namespace atcg