#pragma once

namespace atcg
{
struct DielectricBSDFData
{
    cudaTextureObject_t transmittance_texture;
    cudaTextureObject_t reflectance_texture;
    float ior;
};
}    // namespace atcg