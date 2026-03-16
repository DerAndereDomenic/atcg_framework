#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>

#include <Medium/HomogeneousMediumData.cuh>
#include <Medium/MediumVPtrTable.cuh>

namespace detail
{
template<typename T>
__forceinline__ __device__ T transmittance(float t, T sigma_t)
{
    return glm::exp(-sigma_t * t);
}

template<>
__forceinline__ __device__ atcg::SampledSpectrum transmittance(float t, atcg::SampledSpectrum sigma_t)
{
    return atcg::SampledSpectrum::exp(-sigma_t * t);
}

__forceinline__ __device__ float warp_1d_sample_to_medium_event_distance(float u, float sigma_t)
{
    float t = -glm::log(u) / sigma_t;
    return t;
}

__forceinline__ __device__ float warp_1d_sample_to_medium_event_distance_pdf(float t, float sigma_t)
{
    float pdf = sigma_t * glm::exp(-sigma_t * t);
    return pdf;
}

__forceinline__ __device__ float rgb_to_scalar_weight_max(const glm::vec3& rgb)
{
    return glm::max(glm::max(rgb.x, rgb.y), rgb.z);
}
}    // namespace detail

extern "C" __device__ float __direct_callable__homogeneousMedium_evalTransmittance(const glm::vec3& origin,
                                                                                   const glm::vec3& direction,
                                                                                   float distance,
                                                                                   atcg::PCG32& unused_rng)
{
    const atcg::HomogeneousMediumData* sbt_data =
        *reinterpret_cast<const atcg::HomogeneousMediumData**>(optixGetSbtDataPointer());

    // Evaluate the probability of the light *not* interacting with the medium.
    return detail::transmittance(distance, sbt_data->density);
}

extern "C" __device__ atcg::MediumSamplingResult
__direct_callable__homogeneousMedium_sampleMediumEvent(const glm::vec3& origin,
                                                       const glm::vec3& direction,
                                                       float max_distance,
                                                       const atcg::SampledWavelengths& wavelengths,
                                                       atcg::PCG32& rng)
{
    const atcg::HomogeneousMediumData* sbt_data =
        *reinterpret_cast<const atcg::HomogeneousMediumData**>(optixGetSbtDataPointer());


    // Absorbtion, scattering and extinction coefficients...
    atcg::SampledSpectrum albedo = atcg::SampledSpectrum::fromRGB(sbt_data->albedo, wavelengths);
    atcg::SampledSpectrum Le     = atcg::SampledSpectrum::fromRGB(sbt_data->Le, wavelengths);

    // Scalar projection of scattering coefficient, used to sample the next medium scattering event.
    float sigma_t_scalar = sbt_data->density;


    atcg::MediumSamplingResult result;
    // Dummy implementation:
    // Effectively no medium event.
    result.interaction                    = atcg::MediumInteraction();
    result.interaction.incoming_direction = direction;
    result.transmittance_weight           = atcg::SampledSpectrum(1);
    result.radiance_weight                = atcg::SampledSpectrum(0);

    // Sample the free-flight distance proportional to sigma_s_scalar.
    float sampled_distance = detail::warp_1d_sample_to_medium_event_distance(rng.next1d(), sigma_t_scalar);

    if(sampled_distance < max_distance)
    {
        // Medium event!
        // The sampling succeeded and a scattering event was found at the given distance
        result.interaction.incoming_distance = sampled_distance;
        // Compute the position of the medium interaction as well
        result.interaction.position = origin + result.interaction.incoming_distance * direction;

        // Compute the transmittance including the scattering coeficient sigma_s, divided by sampling probability.
        // float sampling_pdf = warp_1d_sample_to_medium_event_distance_pdf(sampled_distance, sigma_s_scalar);
        // result.transmittance_weight = sbt_data->sigma_s * transmittance(sampled_distance, sigma_t) / sampling_pdf;
        // atcg::SampledSpectrum T =
        //     detail::transmittance(sampled_distance, sigma_t - atcg::SampledSpectrum(sigma_t_scalar));
        // Transmittance will be equal to 1
        result.transmittance_weight = albedo;

        // ? Attenuate by absorption albedo?
        result.radiance_weight = /*(1.0f - sbt_data->sigma_s / sigma_t_scalar) */ Le;
    }
    else
    {
        // No emission, no absorption, no scattering
        // No medium event...
        // The sampling did not succeed, and there is no scattering event *before* the max_distance.
        result.interaction.setInvalid();

        // All no medium events are *the same* event, so we need to compute the transmittance and sampling_pdf for
        // *any* such case, i.e. marginalize over all sampled distances >= max_distance.
        // float sampling_pdf = transmittance(max_distance, sigma_s_scalar);
        // result.transmittance_weight = transmittance(max_distance, sigma_t) / sampling_pdf;
        result.transmittance_weight = atcg::SampledSpectrum(1.0f);
        // detail::transmittance(max_distance, sigma_t - atcg::SampledSpectrum(sigma_t_scalar));
    }

    return result;
}