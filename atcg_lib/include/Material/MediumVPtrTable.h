#pragma once

#include <Core/glm.h>
#include <Material/PhaseFunctionVPtrTable.h>
#include <DataStructure/SampledSpectrum.h>

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
    __device__ float evalTransmittance(const glm::vec3& origin,
                                       const glm::vec3& direction,
                                       float distance,
                                       const glm::mat4& world_to_object,
                                       PCG32& rng) const
    {
        return optixDirectCall<float, const glm::vec3&, const glm::vec3&, float, const glm::mat4&, PCG32&>(
            evalCallIndex,
            origin,
            direction,
            distance,
            world_to_object,
            rng);
    }

    // Sample the position of a new medium event starting at the given (medium) interaction.
    __device__ MediumSamplingResult sampleMediumEvent(const glm::vec3& origin,
                                                      const glm::vec3& direction,
                                                      float max_distance,
                                                      const atcg::SampledWavelengths& wavelengths,
                                                      const glm::mat4& world_to_object,
                                                      PCG32& rng) const
    {
        return optixDirectCall<MediumSamplingResult,
                               const glm::vec3&,
                               const glm::vec3&,
                               float,
                               const atcg::SampledWavelengths&,
                               const glm::mat4&,
                               PCG32&>(sampleCallIndex,
                                       origin,
                                       direction,
                                       max_distance,
                                       wavelengths,
                                       world_to_object,
                                       rng);
    }

#endif    // __CUDACC__
};


struct MediumInstance
{
    const MediumVPtrTable* vptr_table = nullptr;
    glm::mat4 world_to_object         = glm::mat4(1);

#ifdef __CUDACC__
    __device__ float
    evalTransmittance(const glm::vec3& origin, const glm::vec3& direction, float distance, PCG32& rng) const
    {
        return vptr_table->evalTransmittance(origin, direction, distance, world_to_object, rng);
    }

    __device__ MediumSamplingResult sampleMediumEvent(const glm::vec3& origin,
                                                      const glm::vec3& direction,
                                                      float max_distance,
                                                      const atcg::SampledWavelengths& wavelengths,
                                                      PCG32& rng) const
    {
        return vptr_table->sampleMediumEvent(origin, direction, max_distance, wavelengths, world_to_object, rng);
    }
#endif

    __device__ operator bool() const { return vptr_table != nullptr; }
};

}    // namespace atcg