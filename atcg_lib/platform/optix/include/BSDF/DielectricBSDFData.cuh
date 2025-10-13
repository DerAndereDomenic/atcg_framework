#pragma once

#include <DataStructure/CUDATexture.h>

namespace atcg
{
struct DielectricBSDFData
{
    CUDATexture<glm::vec3> diffuse_texture;
    CUDATexture<float> roughness_texture;
    float ior;
};
}    // namespace atcg