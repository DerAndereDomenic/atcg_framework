#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>

#include <Medium/HeterogeneousMediumData.cuh>
#include <Medium/MediumVPtrTable.cuh>

namespace detail
{
__forceinline__ __device__ float warp_1d_sample_to_homogeneous_medium_event_distance(float u, float sigma_t)
{
    float t = -glm::log(u) / sigma_t;
    return t;
}

__device__ float estimate_transmittance_ratio_tracking(const glm::vec3& origin,
                                                       const glm::vec3& direction,
                                                       float max_distance,
                                                       atcg::PCG32& rng)
{
    const atcg::HeterogeneousMediumData* sbt_data =
        *reinterpret_cast<const atcg::HeterogeneousMediumData**>(optixGetSbtDataPointer());

    float distance      = 0;
    float transmittance = 1;
    while(distance < max_distance)
    {
        float step = warp_1d_sample_to_homogeneous_medium_event_distance(rng.next1d(), sbt_data->density_majorant);
        distance += step;
        glm::vec3 step_position = origin + distance * direction;
        float step_density      = sbt_data->density_grid.scale * sbt_data->density_grid.eval(step_position);

        if(distance >= max_distance) break;
        // multiply sigma_n / \bar{sigma}_t
        transmittance *= glm::clamp(1.0f - step_density / sbt_data->density_majorant, 0.0f, 1.0f);
    }
    return transmittance;
}

__device__ float
estimate_transmittance(const glm::vec3& origin, const glm::vec3& direction, float max_distance, atcg::PCG32& rng)
{
    /* Implement:
     * - Evaluate the transmittance over a given distance along the ray, i.e. the transmittance between origin and
     * origin+max_distance*direction.
     * - Implement either the delta-tracking based algorithm or ratio-tracking algorithm.
     * Hint: Use the functions above to sample the medium density and sample distances in homogeneous media.
     */

    //<solution>
    return estimate_transmittance_ratio_tracking(origin, direction, max_distance, rng);
    //</solution>

    return 1;
}

struct DeltaTrackingWeights
{
    float distance;
    atcg::SampledSpectrum transmittance_weight = atcg::SampledSpectrum(1);
    atcg::SampledSpectrum emission_weight      = atcg::SampledSpectrum(0);
};

__device__ DeltaTrackingWeights sample_free_flight_distance_delta_tracking(const glm::vec3& origin,
                                                                           const glm::vec3& direction,
                                                                           float max_distance,
                                                                           const atcg::SampledWavelengths& wavelengths,
                                                                           atcg::PCG32& rng)
{
    const atcg::HeterogeneousMediumData* sbt_data =
        *reinterpret_cast<const atcg::HeterogeneousMediumData**>(optixGetSbtDataPointer());

    DeltaTrackingWeights result;

    /* Implement:
     * - Sample the distance of a medium event in the inverval [0, max_distance] along the given ray in the medium
     * using the delta-tracking algorithm. Hint: Use the functions above to sample the medium density and sample
     * distances in homogeneous media.
     */

    //<solution>
    float distance = 0.0f;
    while(distance < max_distance)
    {
        float step = warp_1d_sample_to_homogeneous_medium_event_distance(rng.next1d(), sbt_data->density_majorant);
        distance += step;
        glm::vec3 step_position = origin + distance * direction;
        float step_density      = sbt_data->density_grid.scale * sbt_data->density_grid.eval(step_position);

        // Russian-roulette-style acceptance of sample.
        if(rng.next1d() < step_density / sbt_data->density_majorant)
        {
            // Scattering or absorbtion event case.
            atcg::SampledSpectrum albedo =
                atcg::SampledSpectrum::fromRGB(sbt_data->albedo_grid.eval(step_position), wavelengths);
            atcg::SampledSpectrum emission =
                atcg::SampledSpectrum::fromRGB(sbt_data->emission_grid.eval(step_position), wavelengths);
            result.transmittance_weight = sbt_data->albedo_grid.scale * albedo;
            result.emission_weight      = sbt_data->emission_grid.scale * emission;
            break;
        }
        else
        {
            // Null-scattering event case.
        }
    }
    // Indicate no medium event if max_distance is exceeded.
    if(distance >= max_distance) distance = std::numeric_limits<float>::signaling_NaN();
    //</solution>

    result.distance = distance;

    return result;
}
}    // namespace detail

extern "C" __device__ glm::vec3 __direct_callable__heterogeneousMedium_evalTransmittance(const glm::vec3& origin,
                                                                                         const glm::vec3& direction,
                                                                                         float distance,
                                                                                         atcg::PCG32& rng)
{
    return glm::vec3(detail::estimate_transmittance(origin, direction, distance, rng));
}

extern "C" __device__ atcg::MediumSamplingResult
__direct_callable__heterogeneousMedium_sampleMediumEvent(const glm::vec3& origin,
                                                         const glm::vec3& direction,
                                                         float max_distance,
                                                         const atcg::SampledWavelengths& wavelengths,
                                                         atcg::PCG32& rng)
{
    const atcg::HeterogeneousMediumData* sbt_data =
        *reinterpret_cast<const atcg::HeterogeneousMediumData**>(optixGetSbtDataPointer());
    // Arbitrarily clamp max_distance.
    // If max_distance would be (close to) infinite, the loop below might not terminate.
    max_distance = glm::clamp(max_distance, 0.0f, 1e6f);

    detail::DeltaTrackingWeights sample =
        detail::sample_free_flight_distance_delta_tracking(origin, direction, max_distance, wavelengths, rng);

    atcg::MediumSamplingResult result;
    // Set incoming ray direction
    result.interaction.incoming_direction = direction;
    result.interaction.incoming_distance  = sample.distance;
    result.transmittance_weight           = sample.transmittance_weight;
    result.radiance_weight                = sample.emission_weight;
    if(!glm::isnan(result.interaction.incoming_distance))
    {
        result.interaction.valid    = true;
        result.interaction.position = origin + result.interaction.incoming_distance * direction;
    }
    return result;
}