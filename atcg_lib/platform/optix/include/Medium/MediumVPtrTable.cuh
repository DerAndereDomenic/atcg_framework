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

#ifdef __CUDACC__

    // Evaluate the transmittance over a certain distance inside of this medium starting at the (medium) interaction.
    __device__ glm::vec3
    evalTransmittance(const glm::vec3& origin, const glm::vec3& direction, float distance, PCG32& rng) const
    {
        return optixDirectCall<glm::vec3, const glm::vec3&, const glm::vec3&, float, PCG32&>(evalCallIndex,
                                                                                             origin,
                                                                                             direction,
                                                                                             distance,
                                                                                             rng);
    }

    // Sample the position of a new medium event starting at the given (medium) interaction.
    __device__ MediumSamplingResult sampleMediumEvent(const glm::vec3& origin,
                                                      const glm::vec3& direction,
                                                      float max_distance,
                                                      PCG32& rng) const
    {
        return optixDirectCall<MediumSamplingResult, const glm::vec3&, const glm::vec3&, float, PCG32&>(sampleCallIndex,
                                                                                                        origin,
                                                                                                        direction,
                                                                                                        max_distance,
                                                                                                        rng);
    }

#endif    // __CUDACC__
};
}    // namespace atcg