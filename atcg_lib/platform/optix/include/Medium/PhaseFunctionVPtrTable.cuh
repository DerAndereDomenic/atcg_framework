#pragma once

#include <Core/glm.h>
#include <Core/SurfaceInteraction.h>
#include <Math/Random.h>
#include <optix.h>
#include <CuDiff/ext/glm.h>
#include <Math/mat6.h>

namespace atcg
{
struct PhaseFunctionSamplingResult
{
    glm::vec3 outgoing_ray_dir;
    float phase_function_weight;
    float sampling_pdf;
};

struct DualPhaseFunctionSamplingResult
{
    CuDiff::Dual<6, glm::vec3> outgoing_ray_dir;
    float phase_function_weight;
    float sampling_pdf;
    atcg::mat6x3 dweight_dx0x1;
};

struct PhaseFunctionEvalResult
{
    float phase_function_value;
    float sampling_pdf;
};

struct PhaseFunctionVPtrTable
{
    uint32_t evalCallIndex;
    uint32_t evalBackwardCallIndex;
    uint32_t sampleCallIndex;
    uint32_t sampleForwardCallIndex;

#ifdef __CUDACC__

    __device__ PhaseFunctionEvalResult evalPhaseFunction(const MediumInteraction& interaction,
                                                         const glm::vec3& outgoing_ray_dir) const
    {
        return optixDirectCall<PhaseFunctionEvalResult, const MediumInteraction&, const glm::vec3&>(evalCallIndex,
                                                                                                    interaction,
                                                                                                    outgoing_ray_dir);
    }

    __device__ void evalPhaseFunctionBackward(const MediumInteraction& interaction,
                                              const glm::vec3& outgoing_ray_dir,
                                              const glm::vec3& output_grad) const
    {
        optixDirectCall<void, const MediumInteraction&, const glm::vec3&, const glm::vec3&>(evalBackwardCallIndex,
                                                                                            interaction,
                                                                                            outgoing_ray_dir,
                                                                                            output_grad);
    }

    __device__ PhaseFunctionSamplingResult samplePhaseFunction(const MediumInteraction& interaction, PCG32& rng) const
    {
        return optixDirectCall<PhaseFunctionSamplingResult, const MediumInteraction&, PCG32&>(sampleCallIndex,
                                                                                              interaction,
                                                                                              rng);
    }

    __device__ DualPhaseFunctionSamplingResult samplePhaseFunctionForward(const DualSurfaceInteraction& interaction,
                                                                          PCG32& rng) const
    {
        return optixDirectCall<DualPhaseFunctionSamplingResult, const DualSurfaceInteraction&, PCG32&>(
            sampleForwardCallIndex,
            interaction,
            rng);
    }

#endif
};
}    // namespace atcg