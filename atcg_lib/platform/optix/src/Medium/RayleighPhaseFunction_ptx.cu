#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>

#include <Medium/PhaseFunctionVPtrTable.cuh>

#include <DataStructure/Frame.h>
#include <Medium/Sampling.h>

extern "C" __device__ atcg::PhaseFunctionEvalResult
__direct_callable__eval_rayleighphase(const atcg::MediumInteraction& interaction, const glm::vec3& outgoing_ray_dir)
{
    atcg::PhaseFunctionEvalResult result;

    atcg::SamplingStrategy<atcg::SamplingStrategyType::RAYLEIGH_PHASE> sampling_strategy;
    result.sampling_pdf         = sampling_strategy.pdf(glm::dot(interaction.incoming_direction, outgoing_ray_dir));
    result.phase_function_value = result.sampling_pdf;
    return result;
}

extern "C" __device__ atcg::PhaseFunctionSamplingResult
__direct_callable__sample_rayleighphase(const atcg::MediumInteraction& interaction, atcg::PCG32& rng)
{
    atcg::SamplingStrategy<atcg::SamplingStrategyType::RAYLEIGH_PHASE> sampling_strategy;
    atcg::Frame local_frame          = atcg::Frame(interaction.incoming_direction);
    glm::vec3 local_outgoing_ray_dir = sampling_strategy.sample(rng.next2d());

    atcg::PhaseFunctionSamplingResult result;
    result.outgoing_ray_dir      = local_frame.toWorld(local_outgoing_ray_dir);
    result.sampling_pdf          = sampling_strategy.pdf(local_outgoing_ray_dir.z);
    result.phase_function_weight = 1.0f;

    return result;
}