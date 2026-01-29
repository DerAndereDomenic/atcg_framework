#pragma once

#include <Core/glm.h>
#include <Core/SurfaceInteraction.h>
#include <Math/Random.h>
#include <optix.h>

namespace atcg
{
struct PhaseFunctionSamplingResult
{
    glm::vec3 outgoing_ray_dir;
    float phase_function_weight;
    float sampling_pdf;
};

struct PhaseFunctionEvalResult
{
    float phase_function_value;
    float sampling_pdf;
};

struct PhaseFunctionVPtrTable
{
    uint32_t evalCallIndex;
    uint32_t sampleCallIndex;

#ifdef __CUDACC__

    __device__ PhaseFunctionEvalResult evalPhaseFunction(const MediumInteraction& interaction,
                                                         const glm::vec3& outgoing_ray_dir) const
    {
        return optixDirectCall<PhaseFunctionEvalResult, const MediumInteraction&, const glm::vec3&>(evalCallIndex,
                                                                                                    interaction,
                                                                                                    outgoing_ray_dir);
    }

    __device__ PhaseFunctionSamplingResult samplePhaseFunction(const MediumInteraction& interaction, PCG32& rng) const
    {
        return optixDirectCall<PhaseFunctionSamplingResult, const MediumInteraction&, PCG32&>(sampleCallIndex,
                                                                                              interaction,
                                                                                              rng);
    }

#endif
};
}    // namespace atcg