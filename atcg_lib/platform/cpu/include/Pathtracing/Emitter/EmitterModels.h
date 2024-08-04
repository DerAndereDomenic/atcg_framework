#pragma once

#include <Pathtracing/CPUInterface.h>
#include <Renderer/Texture.h>
#include <Renderer/Material.h>

namespace atcg
{
class MeshEmitter : public CPUEmitter
{
public:
    MeshEmitter(const torch::Tensor& positions,
                const torch::Tensor& uvs,
                const torch::Tensor& faces,
                const glm::mat4& transform,
                const Material& material);

    ~MeshEmitter();

    virtual glm::vec3 evalLight(const SurfaceInteraction& si) const override;

    virtual EmitterSamplingResult sampleLight(const SurfaceInteraction& si, PCG32& rng) const override;

    virtual PhotonSamplingResult samplePhoton(PCG32& rng) const override;

    virtual float evalLightSamplingPdf(const SurfaceInteraction& last_si, const SurfaceInteraction& si) const override;

private:
    float _emitter_scaling;
    float _total_area;

    torch::Tensor _emissive_texture;

    torch::Tensor _positions;
    torch::Tensor _normals;
    torch::Tensor _uvs;
    torch::Tensor _faces;

    torch::Tensor _mesh_cdf;

    glm::mat4 _local_to_world;
    glm::mat4 _world_to_local;
};

class EnvironmentEmitter : public CPUEmitter
{
public:
    EnvironmentEmitter(const atcg::ref_ptr<atcg::Texture2D>& environment_texture);

    ~EnvironmentEmitter();

    virtual glm::vec3 evalLight(const SurfaceInteraction& si) const override;

    virtual EmitterSamplingResult sampleLight(const SurfaceInteraction& si, PCG32& rng) const override;

    virtual PhotonSamplingResult samplePhoton(PCG32& rng) const override;

    virtual float evalLightSamplingPdf(const SurfaceInteraction& last_si, const SurfaceInteraction& si) const override;

private:
    torch::Tensor _environment_texture;
};
}    // namespace atcg