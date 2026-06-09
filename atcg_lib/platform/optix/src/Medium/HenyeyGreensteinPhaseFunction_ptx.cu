#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>
#include <Core/GlobalAtomicAdd.h>

#include <Math/Random.h>

#include <Math/Functions.h>
#include <Medium/PhaseFunctionVPtrTable.cuh>
#include <Medium/HenyeyGreensteinPhaseFunctionData.cuh>

#include <DataStructure/Frame.h>
#include <Medium/Sampling.h>
#include <CuDiff/CuDiff.h>
#include <CuDiff/ext/glm.h>

extern "C" __device__ atcg::PhaseFunctionEvalResult
__direct_callable__eval_hgphase(const atcg::MediumInteraction& interaction, const glm::vec3& outgoing_ray_dir)
{
    const atcg::HenyeyGreensteinPhaseFunctionData* sbt_data =
        *reinterpret_cast<const atcg::HenyeyGreensteinPhaseFunctionData**>(optixGetSbtDataPointer());

    float g = *(sbt_data->g);

    atcg::PhaseFunctionEvalResult result;
    // Since we can sample the phase function exactly, the sampling pdf is equal to the phase function itself.
    // The difference is that the phase function is in general allowed to return a "chromatic" value, and the sampling
    // pdf returns a scalar value.
    atcg::SamplingStrategy<atcg::SamplingStrategyType::HG_PHASE> sampling_strategy(g);
    result.sampling_pdf         = sampling_strategy.pdf(glm::dot(interaction.incoming_direction, outgoing_ray_dir));
    result.phase_function_value = result.sampling_pdf;
    return result;
}

extern "C" __device__ atcg::PhaseFunctionSamplingResult
__direct_callable__sample_hgphase(const atcg::MediumInteraction& interaction, atcg::PCG32& rng)
{
    const atcg::HenyeyGreensteinPhaseFunctionData* sbt_data =
        *reinterpret_cast<const atcg::HenyeyGreensteinPhaseFunctionData**>(optixGetSbtDataPointer());

    float g = *(sbt_data->g);

    atcg::SamplingStrategy<atcg::SamplingStrategyType::HG_PHASE> sampling_strategy(g);

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

extern "C" __device__ atcg::DualPhaseFunctionSamplingResult
__direct_callable__sample_hgphase_forward(const atcg::DualMediumInteraction& interaction, atcg::PCG32& rng)
{
    const atcg::HenyeyGreensteinPhaseFunctionData* sbt_data =
        *reinterpret_cast<const atcg::HenyeyGreensteinPhaseFunctionData**>(optixGetSbtDataPointer());

    float g = *(sbt_data->g);

    atcg::SamplingStrategy<atcg::SamplingStrategyType::HG_PHASE> sampling_strategy(g);

    atcg::Frame local_frame          = atcg::Frame(interaction.incoming_direction);
    glm::vec3 local_outgoing_ray_dir = sampling_strategy.sample(rng.next2d());

    atcg::DualPhaseFunctionSamplingResult result;
    result.outgoing_ray_dir = local_frame.toWorld(local_outgoing_ray_dir);
    result.sampling_pdf     = sampling_strategy.pdf(local_outgoing_ray_dir.z);
    // result.phase_function_weight = glm::vec3(henyey_greenstein_phase_function(local_outgoing_ray_dir.z, sbt_data->g))
    // / result.sampling_pdf;
    result.phase_function_weight = 1.0f;
    result.dphase_dx0x1          = atcg::mat6x3(0.0f);

    return result;
}

extern "C" __device__ void __direct_callable__sample_hgphase_backward(const atcg::MediumInteraction& interaction,
                                                                      atcg::PCG32& rng,
                                                                      const glm::vec3& dL_dweight,
                                                                      const glm::vec3& dL_dwo)
{
    const atcg::HenyeyGreensteinPhaseFunctionData* sbt_data =
        *reinterpret_cast<const atcg::HenyeyGreensteinPhaseFunctionData**>(optixGetSbtDataPointer());

    if(!sbt_data->optimize_g)
    {
        return;
    }

    float g_ = *(sbt_data->g);

    CuDiff::Dual<1, float> g(g_);
    g.setDerivative(0, 1.0f);

    atcg::SamplingStrategy<atcg::SamplingStrategyType::HG_PHASE, CuDiff::Dual<1, float>> sampling_strategy(g);

    atcg::Frame local_frame     = atcg::Frame(interaction.incoming_direction);
    auto local_outgoing_ray_dir = sampling_strategy.sample(rng.next2d());

    auto outgoing_ray_dir = local_frame.toWorld(local_outgoing_ray_dir);

    // dweight_dg = 0 because of perfect importance sampling, the weight is constant

    glm::vec3 dwo_dg = outgoing_ray_dir.derivative(0);

    float g_grad = glm::dot(dL_dwo, dwo_dg);

    if(isfinite(g_grad))
    {
        atcg::globalAtomicAdd(sbt_data->g_grad, g_grad);
    }
}

extern "C" __device__ atcg::DualPhaseFunctionEvalResult
__direct_callable__eval_hgphase_forward(const atcg::DualMediumInteraction& interaction,
                                        const CuDiff::Dual<6, glm::vec3>& outgoing_ray_dir)
{
    const atcg::HenyeyGreensteinPhaseFunctionData* sbt_data =
        *reinterpret_cast<const atcg::HenyeyGreensteinPhaseFunctionData**>(optixGetSbtDataPointer());

    float g = *(sbt_data->g);

    atcg::DualPhaseFunctionEvalResult result;
    // Since we can sample the phase function exactly, the sampling pdf is equal to the phase function itself.
    // The difference is that the phase function is in general allowed to return a "chromatic" value, and the sampling
    // pdf returns a scalar value.
    atcg::SamplingStrategy<atcg::SamplingStrategyType::HG_PHASE> sampling_strategy(g);
    auto phase_value            = sampling_strategy.pdf(CuDiff::dot(interaction.incoming_direction, outgoing_ray_dir));
    result.sampling_pdf         = phase_value.val();
    result.phase_function_value = result.sampling_pdf;

    glm::mat3 Jphase_dx0 = glm::mat3(glm::vec3(phase_value.derivative(0)),
                                     glm::vec3(phase_value.derivative(1)),
                                     glm::vec3(phase_value.derivative(2)));
    glm::mat3 Jphase_dx1 = glm::mat3(glm::vec3(phase_value.derivative(3)),
                                     glm::vec3(phase_value.derivative(4)),
                                     glm::vec3(phase_value.derivative(5)));
    result.dphase_dx0x1  = atcg::mat6x3(Jphase_dx0, Jphase_dx1);
    return result;
}

extern "C" __device__ void __direct_callable__eval_hgphase_backward(const atcg::MediumInteraction& interaction,
                                                                    const glm::vec3& outgoing_ray_dir,
                                                                    const glm::vec3& output_grad)
{
    const atcg::HenyeyGreensteinPhaseFunctionData* sbt_data =
        *reinterpret_cast<const atcg::HenyeyGreensteinPhaseFunctionData**>(optixGetSbtDataPointer());

    if(!sbt_data->optimize_g)
    {
        return;
    }

    float g = *(sbt_data->g);

    float cos_theta = glm::dot(interaction.incoming_direction, outgoing_ray_dir);
    atcg::SamplingStrategy<atcg::SamplingStrategyType::HG_PHASE> sampling_strategy(g);
    float phase_value = sampling_strategy.pdf(cos_theta);

    // dphase_dg
    float g2        = g * g;
    float denom     = 1 + g2 - 2 * g * cos_theta;
    float dphase_dg = (-2.0f * g * glm::pow(denom, -1.5f) -
                       1.5f * (1.0f - g2) * (2.0f * g - 2.0f * cos_theta) * glm::pow(denom, -2.5f)) /
                      (4.0f * glm::pi<float>());


    // We divide by the phase value here because output_grad still contains the phase value
    float g_grad = glm::dot(glm::vec3(1.0f), dphase_dg * output_grad / phase_value);

    if(isfinite(g_grad))
    {
        atcg::globalAtomicAdd(sbt_data->g_grad, g_grad);
    }
}