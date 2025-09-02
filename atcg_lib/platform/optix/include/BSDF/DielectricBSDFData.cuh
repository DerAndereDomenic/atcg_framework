#pragma once

namespace atcg
{
struct DielectricBSDFData
{
    cudaTextureObject_t diffuse_texture;
    cudaTextureObject_t roughness_texture;
    float ior;
};
}    // namespace atcg