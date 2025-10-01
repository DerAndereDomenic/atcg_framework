#pragma once

#include <DataStructure/TextureSampler.h>

namespace atcg
{
struct PBRBSDFData
{
    TextureSampler<glm::vec3> diffuse_texture;
    TextureSampler<float> metallic_texture;
    TextureSampler<float> roughness_texture;

    TextureSampler<glm::vec3> diffuse_grad;
    TextureSampler<float> metallic_grad;
    TextureSampler<float> roughness_grad;
};
}    // namespace atcg