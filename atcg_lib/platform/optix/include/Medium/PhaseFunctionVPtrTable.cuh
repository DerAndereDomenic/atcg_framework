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
    float phase_function_weight = 0.0f;
    float sampling_pdf          = 0.0f;
};

struct DualPhaseFunctionSamplingResult
{
    CuDiff::Dual<6, glm::vec3> outgoing_ray_dir;
    float phase_function_weight = 0.0f;
    float sampling_pdf          = 0.0f;
    atcg::mat6x3 dphase_dx0x1   = atcg::mat6x3(0.0f);
};

struct PhaseFunctionEvalResult
{
    float phase_function_value = 0.0f;
    float sampling_pdf         = 0.0f;
};

struct DualPhaseFunctionEvalResult
{
    float phase_function_value = 0.0f;
    float sampling_pdf         = 0.0f;
    atcg::mat6x3 dphase_dx0x1  = atcg::mat6x3(0.0f);
};

struct PhaseFunctionVPtrTable
{
    uint32_t evalCallIndex;
    uint32_t evalForwardCallIndex;
    uint32_t evalBackwardCallIndex;
    uint32_t sampleCallIndex;
    uint32_t sampleForwardCallIndex;
    uint32_t sampleBackwardCallIndex;

#ifdef __CUDACC__

    __device__ PhaseFunctionEvalResult evalPhaseFunction(const MediumInteraction& interaction,
                                                         const glm::vec3& outgoing_ray_dir) const
    {
        return optixDirectCall<PhaseFunctionEvalResult, const MediumInteraction&, const glm::vec3&>(evalCallIndex,
                                                                                                    interaction,
                                                                                                    outgoing_ray_dir);
    }

    __device__ DualPhaseFunctionEvalResult
    evalPhaseFunctionForward(const DualMediumInteraction& interaction,
                             const CuDiff::Dual<6, glm::vec3>& outgoing_ray_dir) const
    {
        return optixDirectCall<DualPhaseFunctionEvalResult,
                               const DualMediumInteraction&,
                               const CuDiff::Dual<6, glm::vec3>&>(evalForwardCallIndex, interaction, outgoing_ray_dir);
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

    __device__ DualPhaseFunctionSamplingResult samplePhaseFunctionForward(const DualMediumInteraction& interaction,
                                                                          PCG32& rng) const
    {
        return optixDirectCall<DualPhaseFunctionSamplingResult, const DualMediumInteraction&, PCG32&>(
            sampleForwardCallIndex,
            interaction,
            rng);
    }

    __device__ void samplePhaseFunctionBackward(const MediumInteraction& interaction,
                                                PCG32& rng,
                                                const glm::vec3& dL_dweight,
                                                const glm::vec3& dL_dwo) const
    {
        return optixDirectCall<void, const MediumInteraction&, PCG32&, const glm::vec3&, const glm::vec3&>(
            sampleBackwardCallIndex,
            interaction,
            rng,
            dL_dweight,
            dL_dwo);
    }

#endif
};
}    // namespace atcg