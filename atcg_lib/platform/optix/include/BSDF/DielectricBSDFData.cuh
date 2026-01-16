#pragma once

#include <DataStructure/TextureSampler.h>

namespace atcg
{
struct DielectricBSDFData
{
    TextureSampler<glm::vec3> diffuse_texture;
    TextureSampler<float> roughness_texture;
    TextureSampler<float> ior_texture;

    TextureSampler<glm::vec3> diffuse_grad;
    TextureSampler<float> roughness_grad;
    TextureSampler<float> ior_grad;

    bool optimizable;
};
}    // namespace atcg