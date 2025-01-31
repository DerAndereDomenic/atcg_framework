#pragma once

namespace atcg
{
struct PBRBSDFData
{
    cudaTextureObject_t diffuse_texture;
    cudaTextureObject_t metallic_texture;
    cudaTextureObject_t roughness_texture;
};
}    // namespace atcg