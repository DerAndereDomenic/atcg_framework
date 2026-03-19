#pragma once

#include <Core/glm.h>
#include <Medium/PhaseFunctionVPtrTable.cuh>
#include <Spectrum/SampledSpectrum.h>

namespace atcg
{
struct MediumSamplingResult
{
    MediumInteraction interaction;
    SampledSpectrum transmittance_weight;
    SampledSpectrum radiance_weight;
};

struct MediumVPtrTable
{
    const PhaseFunctionVPtrTable* phase_function;

    uint32_t evalCallIndex;
    uint32_t sampleCallIndex;
    uint32_t sampleBackwardCallIndex;
    uint32_t evalTransmittanceBackwardCallIndex;

#ifdef __CUDACC__

    // Evaluate the transmittance over a certain distance inside of this medium starting at the (medium) interaction.
    __device__ float
    evalTransmittance(const glm::vec3& origin, const glm::vec3& direction, float distance, PCG32& rng) const
    {
        return optixDirectCall<float, const glm::vec3&, const glm::vec3&, float, PCG32&>(evalCallIndex,
                                                                                         origin,
                                                                                         direction,
                                                                                         distance,
                                                                                         rng);
    }

    // Sample the position of a new medium event starting at the given (medium) interaction.
    __device__ MediumSamplingResult sampleMediumEvent(const glm::vec3& origin,
                                                      const glm::vec3& direction,
                                                      float max_distance,
                                                      const atcg::SampledWavelengths& wavelengths,
                                                      PCG32& rng) const
    {
        return optixDirectCall<MediumSamplingResult,
                               const glm::vec3&,
                               const glm::vec3&,
                               float,
                               const atcg::SampledWavelengths&,
                               PCG32&>(sampleCallIndex, origin, direction, max_distance, wavelengths, rng);
    }

    __device__ void sampleMediumEventBackward(const glm::vec3& origin,
                                              const glm::vec3& direction,
                                              float max_distance,
                                              const atcg::SampledWavelengths& wavelengths,
                                              PCG32& rng,
                                              const glm::vec3& out_grad) const
    {
        optixDirectCall<void,
                        const glm::vec3&,
                        const glm::vec3&,
                        float,
                        const atcg::SampledWavelengths&,
                        PCG32&,
                        const glm::vec3&>(sampleBackwardCallIndex,
                                          origin,
                                          direction,
                                          max_distance,
                                          wavelengths,
                                          rng,
                                          out_grad);
    }

    __device__ void evalTransmittanceBackward(const glm::vec3& origin,
                                              const glm::vec3& direction,
                                              float distance,
                                              PCG32& rng,
                                              const glm::vec3& out_grad) const
    {
        optixDirectCall<void, const glm::vec3&, const glm::vec3&, float, PCG32&, const glm::vec3&>(
            evalTransmittanceBackwardCallIndex,
            origin,
            direction,
            distance,
            rng,
            out_grad);
    }

#endif    // __CUDACC__
};
}    // namespace atcg