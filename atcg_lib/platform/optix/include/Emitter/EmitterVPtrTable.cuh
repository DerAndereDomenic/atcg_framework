#pragma once

#include <Core/glm.h>
#include <Core/SurfaceInteraction.h>
#include <Math/Random.h>
#include <Emitter/EmitterFlags.h>
#include <Spectrum/SampledSpectrum.h>
#include <CuDiff/CuDiff.h>

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

struct DualEmitterSamplingResult
{
    CuDiff::Dual<6, glm::vec3> direction_to_light;
    float distance_to_light;
    CuDiff::Dual<6, glm::vec3> radiance_weight_at_receiver;
    float sampling_pdf;
};

struct PhotonSamplingResult
{
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 normal;
    SampledSpectrum radiance_weight;    // Le / p in area measure
    float pdf;                          // 1/Area
    glm::vec3 uvs;
};

struct EdgeSamplingResult
{
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 normal;
    SampledSpectrum radiance_weight;
    float pdf;
};

struct EmitterVPtrTable
{
    EmitterFlags flags;

    uint32_t evalCallIndex;
    uint32_t evalForwardCallIndex;
    uint32_t sampleCallIndex;
    uint32_t sampleForwardCallIndex;
    uint32_t evalPdfCallIndex;
    uint32_t sampleEdgeCallIndex;

#ifdef __CUDACC__

    __device__ SampledSpectrum evalLight(const SurfaceInteraction& si,
                                         const atcg::SampledWavelengths& wavelengths) const
    {
        return optixDirectCall<SampledSpectrum, const SurfaceInteraction&, const atcg::SampledWavelengths&>(
            evalCallIndex,
            si,
            wavelengths);
    }

    __device__ CuDiff::Dual<6, glm::vec3> evalLightForward(const DualSurfaceInteraction& si,
                                                           const atcg::SampledWavelengths& wavelengths) const
    {
        return optixDirectCall<CuDiff::Dual<6, glm::vec3>,
                               const DualSurfaceInteraction&,
                               const atcg::SampledWavelengths&>(evalForwardCallIndex, si, wavelengths);
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

    __device__ DualEmitterSamplingResult sampleLightForward(const DualSurfaceInteraction& si,
                                                            const atcg::SampledWavelengths& wavelengths,
                                                            PCG32& rng) const
    {
        return optixDirectCall<DualEmitterSamplingResult,
                               const DualSurfaceInteraction&,
                               const atcg::SampledWavelengths&,
                               PCG32&>(sampleForwardCallIndex, si, wavelengths, rng);
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

#endif
};
}    // namespace atcg