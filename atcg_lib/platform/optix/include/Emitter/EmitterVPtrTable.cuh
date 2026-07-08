#pragma once

#include <Core/glm.h>
#include <Core/SurfaceInteraction.h>
#include <Math/Random.h>
#include <Emitter/EmitterFlags.h>
#include <Spectrum/SampledSpectrum.h>

#include <optix.h>

namespace atcg
{
struct EmitterSamplingResult
{
    glm::vec3 direction_to_light;
    float distance_to_light;
    glm::vec3 normal_at_light;
    SampledSpectrum radiance_weight_at_receiver;
    float sampling_pdf;
    glm::vec3 uvs;
};

struct PhotonSamplingResult
{
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 normal;
    SampledSpectrum radiance_weight;
    float pdf;
    glm::vec3 uvs;
};

struct EmitterVPtrTable
{
    EmitterFlags flags = EmitterFlags::None;

    uint32_t evalCallIndex;
    uint32_t sampleCallIndex;
    uint32_t evalPdfCallIndex;
    uint32_t samplePhotonCallIndex;

#ifdef __CUDACC__

    __device__ SampledSpectrum evalLight(const SurfaceInteraction& si,
                                         const atcg::SampledWavelengths& wavelengths) const
    {
        return optixDirectCall<SampledSpectrum, const SurfaceInteraction&, const atcg::SampledWavelengths&>(
            evalCallIndex,
            si,
            wavelengths);
    }

    __device__ EmitterSamplingResult sampleLight(const AnyInteraction& si,
                                                 const atcg::SampledWavelengths& wavelengths,
                                                 PCG32& rng) const
    {
        return optixDirectCall<EmitterSamplingResult, const AnyInteraction&, const atcg::SampledWavelengths&, PCG32&>(
            sampleCallIndex,
            si,
            wavelengths,
            rng);
    }

    __device__ float evalLightSamplingPdf(const AnyInteraction& last_si, const SurfaceInteraction& si) const
    {
        return optixDirectCall<float, const AnyInteraction&, const SurfaceInteraction&>(evalPdfCallIndex, last_si, si);
    }

    __device__ PhotonSamplingResult samplePhoton(const atcg::SampledWavelengths& wavelengths, PCG32& rng) const
    {
        return optixDirectCall<PhotonSamplingResult, const atcg::SampledWavelengths&, PCG32&>(samplePhotonCallIndex,
                                                                                              wavelengths,
                                                                                              rng);
    }

#endif
};
}    // namespace atcg