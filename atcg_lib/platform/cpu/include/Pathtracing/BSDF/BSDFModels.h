#pragma once

#include <Pathtracing/CPUInterface.h>
#include <Renderer/Material.h>

namespace atcg
{
class PBRBSDF : public CPUBSDF
{
public:
    PBRBSDF(const Material& material);

    ~PBRBSDF();

    virtual BSDFSamplingResult sampleBSDF(const SurfaceInteraction& si, PCG32& rng) const override;

    virtual BSDFEvalResult evalBSDF(const SurfaceInteraction& si, const glm::vec3& outgoing_dir) override;

private:
    torch::Tensor _diffuse_image;
    torch::Tensor _roughness_image;
    torch::Tensor _metallic_image;
};

class RefractiveBSDF : public CPUBSDF
{
public:
    RefractiveBSDF(const Material& material);

    ~RefractiveBSDF();

    virtual BSDFSamplingResult sampleBSDF(const SurfaceInteraction& si, PCG32& rng) const override;

    virtual BSDFEvalResult evalBSDF(const SurfaceInteraction& si, const glm::vec3& outgoing_dir) override;

private:
    float _ior;
    torch::Tensor _diffuse_image;
};

}    // namespace atcg