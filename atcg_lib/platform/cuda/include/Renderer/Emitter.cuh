#pragma once

#include <Math/Random.h>
#include <Math/SurfaceInteraction.h>

#ifdef ATCG_CUDA_BACKEND
    #include <optix.h>
#endif

namespace atcg
{
struct EmitterSamplingResult
{
    float distance_to_light;
    glm::vec3 normal_at_light;
    glm::vec3 radiance_weight_at_receiver;
    float sampling_pdf;
};

// TODO: Make this dependent on a shape instance and don't copy the data twice
struct MeshEmitterData
{
    glm::vec3* positions;
    glm::vec3* normals;
    glm::vec3* uvs;
    glm::u32vec3* faces;

    float emitter_scaling;
    cudaTextureObject_t emissive_texture;
};

struct EnvironmentEmitterData
{
    cudaTextureObject_t environment_texture;
};

struct EmitterVPtrTable
{
    uint32_t evalCallIndex;
    uint32_t sampleCallIndex;

#ifdef __CUDACC__

    __device__ glm::vec3 evalLight(const SurfaceInteraction& si) const
    {
        return optixDirectCall<glm::vec3, const SurfaceInteraction&>(evalCallIndex, si);
    }

    __device__ EmitterSamplingResult sampleLight(const SurfaceInteraction& si, PCG32& rng) const
    {
        return optixDirectCall<EmitterSamplingResult, const SurfaceInteraction&, PCG32&>(sampleCallIndex, si, rng);
    }

#endif
};
}    // namespace atcg