#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>

#include <Math/Functions.h>
#include <Medium/PhaseFunctionVPtrTable.cuh>
#include <Medium/HenyeyGreensteinPhaseFunctionData.cuh>

#include <DataStructure/Frame.h>
#include <Medium/Sampling.h>


extern "C" __device__ atcg::PhaseFunctionEvalResult
__direct_callable__eval_hgphase(const atcg::MediumInteraction& interaction, const glm::vec3& outgoing_ray_dir)
{
    const atcg::HenyeyGreensteinPhaseFunctionData* sbt_data =
        *reinterpret_cast<const atcg::HenyeyGreensteinPhaseFunctionData**>(optixGetSbtDataPointer());

    atcg::PhaseFunctionEvalResult result;
    // Since we can sample the phase function exactly, the sampling pdf is equal to the phase function itself.
    // The difference is that the phase function is in general allowed to return a "chromatic" value, and the sampling
    // pdf returns a scalar value.
    atcg::SamplingStrategy<atcg::SamplingStrategyType::HG_PHASE> sampling_strategy(sbt_data->g);
    result.sampling_pdf         = sampling_strategy.pdf(glm::dot(interaction.incoming_direction, outgoing_ray_dir));
    result.phase_function_value = result.sampling_pdf;
    return result;
}

extern "C" __device__ atcg::PhaseFunctionSamplingResult
__direct_callable__sample_hgphase(const atcg::MediumInteraction& interaction, atcg::PCG32& rng)
{
    const atcg::HenyeyGreensteinPhaseFunctionData* sbt_data =
        *reinterpret_cast<const atcg::HenyeyGreensteinPhaseFunctionData**>(optixGetSbtDataPointer());

    atcg::SamplingStrategy<atcg::SamplingStrategyType::HG_PHASE> sampling_strategy(sbt_data->g);

    atcg::Frame local_frame          = atcg::Frame(interaction.incoming_direction);
    glm::vec3 local_outgoing_ray_dir = sampling_strategy.sample(rng.next2d());

    atcg::PhaseFunctionSamplingResult result;
    result.outgoing_ray_dir = local_frame.toWorld(local_outgoing_ray_dir);
    result.sampling_pdf     = sampling_strategy.pdf(local_outgoing_ray_dir.z);
    // result.phase_function_weight = glm::vec3(henyey_greenstein_phase_function(local_outgoing_ray_dir.z, sbt_data->g))
    // / result.sampling_pdf;
    result.phase_function_weight = 1.0f;

    return result;
}