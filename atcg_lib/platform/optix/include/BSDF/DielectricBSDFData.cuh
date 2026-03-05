#pragma once

#include <DataStructure/TextureSampler.h>

namespace atcg
{
struct DielectricBSDFData
{
    TextureSampler<glm::vec3> diffuse_texture;
    TextureSampler<float> roughness_texture;
    TextureSampler<float> ior_texture;

    TextureWriter<glm::vec3> diffuse_grad;
    TextureWriter<float> roughness_grad;
    TextureWriter<float> ior_grad;

    bool optimizable;
};
}    // namespace atcg