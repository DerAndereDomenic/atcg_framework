#pragma once

#include <DataStructure/CUDATexture.h>

namespace atcg
{
struct DielectricBSDFData
{
    CUDATexture<glm::vec3> diffuse_texture;
    CUDATexture<float> roughness_texture;
    CUDATexture<float> ior_texture;
};
}    // namespace atcg