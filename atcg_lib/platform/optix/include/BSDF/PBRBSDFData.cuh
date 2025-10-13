#pragma once

#include <DataStructure/CUDATexture.h>

namespace atcg
{
struct PBRBSDFData
{
    CUDATexture<glm::vec3> diffuse_texture;
    CUDATexture<float> metallic_texture;
    CUDATexture<float> roughness_texture;
};
}    // namespace atcg