#pragma once

#include <Core/glm.h>
#include <Core/SurfaceInteraction.h>
#include <Math/Random.h>
#include <Emitter/EmitterFlags.h>
#include <Spectrum/SampledSpectrum.h>
#include <CuDiff/CuDiff.h>
#include <Math/mat6.h>

#include <optix.h>

namespace atcg
{
struct EmitterSamplingResult
{
    glm::vec3 direction_to_light;
    float distance_to_light;
    glm::vec3 normal_at_light;
    SampledSpectrum radiance_weight_at_receiver = SampledSpectrum(0.0f);
    float sampling_pdf                          = 0.0f;
    glm::vec3 uvs;
};

struct EmitterDualSamplingResult
{
    CuDiff::Dual<6, glm::vec3> direction_to_light;
    float distance_to_light;
    glm::vec3 radiance_weight_at_receiver = glm::vec3(0.0f);
    atcg::mat6x3 dLe_dx0x1                = atcg::mat6x3(0.0f);
    float sampling_pdf                    = 0.0f;
};

struct EmitterDualEvalResult
{
    glm::vec3 radiance_weight_at_receiver = glm::vec3(0.0f);
    atcg::mat6x3 dLe_dx0x1                = atcg::mat6x3(0.0f);
};

struct PhotonSamplingResult
{
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 normal;
    SampledSpectrum radiance_weight = SampledSpectrum(0.0f);
    float pdf                       = 0.0f;
    glm::vec3 uvs;
};

struct EdgeSamplingResult
{
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 normal;
    SampledSpectrum radiance_weight = SampledSpectrum(0.0f);
    float pdf                       = 0.0f;
};

struct EmitterVPtrTable
{
    EmitterFlags flags = EmitterFlags::None;

    uint32_t evalCallIndex;
    uint32_t evalForwardCallIndex;
    uint32_t sampleCallIndex;
    uint32_t sampleForwardCallIndex;
    uint32_t evalPdfCallIndex;
    uint32_t sampleEdgeCallIndex;
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

    __device__ EmitterDualEvalResult evalLightForward(const DualSurfaceInteraction& si,
                                                      const atcg::SampledWavelengths& wavelengths) const
    {
        return optixDirectCall<EmitterDualEvalResult, const DualSurfaceInteraction&, const atcg::SampledWavelengths&>(
            evalForwardCallIndex,
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

    __device__ EmitterDualSamplingResult sampleLightForward(const AnyDualInteraction& ai,
                                                            const atcg::SampledWavelengths& wavelengths,
                                                            PCG32& rng) const
    {
        return optixDirectCall<EmitterDualSamplingResult,
                               const AnyDualInteraction&,
                               const atcg::SampledWavelengths&,
                               PCG32&>(sampleForwardCallIndex, ai, wavelengths, rng);
    }

    __device__ EdgeSamplingResult sampleEdge(const SurfaceInteraction& si,
                                             const atcg::SampledWavelengths& wavelengths,
                                             PCG32& rng) const
    {
        return optixDirectCall<EdgeSamplingResult, const SurfaceInteraction&, const atcg::SampledWavelengths&, PCG32&>(
            sampleEdgeCallIndex,
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