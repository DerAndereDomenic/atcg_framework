#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>

#include <Math/Functions.h>
#include <Medium/PhaseFunctionVPtrTable.cuh>
#include <Medium/HenyeyGreensteinPhaseFunctionData.cuh>

namespace detail
{

__device__ float henyey_greenstein_phase_function(float cos_theta, float g)
{
    float g2    = g * g;
    float area  = 4 * glm::pi<float>();    // area of sphere
    float phase = (1 - g2) / area * glm::pow((1 + g2 - 2 * g * cos_theta), -1.5f);
    return phase;
}

__device__ glm::vec3 warp_square_to_sphere_henyey_greenstein(const glm::vec2& uv, float g)
{
    float u1 = uv.x;
    float u2 = uv.y;

    float g2        = g * g;
    float d         = (1 - g2) / (1 - g + 2 * g * u1);
    float cos_theta = 0.5 / g * (1 + g2 - d * d);

    float sin_theta = glm::sqrt(glm::max(0.0f, 1.0f - cos_theta * cos_theta));
    float phi       = 2 * glm::pi<float>() * u2;

    float x = sin_theta * glm::cos(phi);
    float y = sin_theta * glm::sin(phi);
    float z = cos_theta;

    return glm::vec3(x, y, z);
}

__device__ float warp_square_to_sphere_henyey_greenstein_pdf(const glm::vec3& result, float g)
{
    return henyey_greenstein_phase_function(result.z, g);
}
}    // namespace detail


extern "C" __device__ atcg::PhaseFunctionEvalResult
__direct_callable__eval_hgphase(const atcg::MediumInteraction& interaction, const glm::vec3& outgoing_ray_dir)
{
    const atcg::HenyeyGreensteinPhaseFunctionData* sbt_data =
        *reinterpret_cast<const atcg::HenyeyGreensteinPhaseFunctionData**>(optixGetSbtDataPointer());

    atcg::PhaseFunctionEvalResult result;
    // Since we can sample the phase function exactly, the sampling pdf is equal to the phase function itself.
    // The difference is that the phase function is in general allowed to return a "chromatic" value, and the sampling
    // pdf returns a scalar value.
    result.sampling_pdf =
        detail::henyey_greenstein_phase_function(glm::dot(interaction.incoming_direction, outgoing_ray_dir),
                                                 sbt_data->g);
    result.phase_function_value = glm::vec3(result.sampling_pdf);
    return result;
}

extern "C" __device__ atcg::PhaseFunctionSamplingResult
__direct_callable__sample_hgphase(const atcg::MediumInteraction& interaction, atcg::PCG32& rng)
{
    const atcg::HenyeyGreensteinPhaseFunctionData* sbt_data =
        *reinterpret_cast<const atcg::HenyeyGreensteinPhaseFunctionData**>(optixGetSbtDataPointer());

    glm::mat3 local_frame            = atcg::Math::compute_local_frame(interaction.incoming_direction);
    glm::vec3 local_outgoing_ray_dir = detail::warp_square_to_sphere_henyey_greenstein(rng.next2d(), sbt_data->g);

    atcg::PhaseFunctionSamplingResult result;
    result.outgoing_ray_dir = local_frame * local_outgoing_ray_dir;
    result.sampling_pdf     = detail::warp_square_to_sphere_henyey_greenstein_pdf(local_outgoing_ray_dir, sbt_data->g);
    // result.phase_function_weight = glm::vec3(henyey_greenstein_phase_function(local_outgoing_ray_dir.z, sbt_data->g))
    // / result.sampling_pdf;
    result.phase_function_weight = glm::vec3(1);

    return result;
}