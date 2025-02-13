#pragma once

#include <Core/glm.h>
#include <Core/SurfaceInteraction.h>
#include <Math/Random.h>
#include <Emitter/EmitterFlags.h>

#include <optix.h>

namespace atcg
{
struct EmitterSamplingResult
{
    glm::vec3 direction_to_light;
    float distance_to_light;
    glm::vec3 normal_at_light;
    glm::vec3 radiance_weight_at_receiver;
    float sampling_pdf;
    glm::vec3 uvs;
};

struct PhotonSamplingResult
{
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 normal;
    glm::vec3 radiance_weight;    // Le / p in area measure
    float pdf;                    // 1/Area
    glm::vec3 uvs;
};

struct EmitterVPtrTable
{
    EmitterFlags flags;

    uint32_t evalCallIndex;
    uint32_t sampleCallIndex;
    uint32_t evalPdfCallIndex;

#ifdef __CUDACC__

    __device__ glm::vec3 evalLight(const SurfaceInteraction& si) const
    {
        return optixDirectCall<glm::vec3, const SurfaceInteraction&>(evalCallIndex, si);
    }

    __device__ EmitterSamplingResult sampleLight(const SurfaceInteraction& si, PCG32& rng) const
    {
        return optixDirectCall<EmitterSamplingResult, const SurfaceInteraction&, PCG32&>(sampleCallIndex, si, rng);
    }

    __device__ float evalLightSamplingPdf(const SurfaceInteraction& last_si, const SurfaceInteraction& si) const
    {
        return optixDirectCall<float, const SurfaceInteraction&, const SurfaceInteraction&>(evalPdfCallIndex,
                                                                                            last_si,
                                                                                            si);
    }

#endif
};
}    // namespace atcg