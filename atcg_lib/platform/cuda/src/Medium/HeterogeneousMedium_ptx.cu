#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>

#include <Material/HeterogeneousMediumData.h>
#include <Material/MediumVPtrTable.h>
#include <Medium/Transmittance.h>

namespace detail
{

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
    atcg::SamplingStrategy<atcg::SamplingStrategyType::EXPONENTIAL_SAMPLING> sampling_strategy(
        sbt_data->density_majorant);
    while(distance < max_distance)
    {
        float step = sampling_strategy.sample(rng.next1d());
        distance += step;
        glm::vec3 step_position = origin + distance * direction;
        float step_density      = sbt_data->density_grid.eval(step_position);

        // Russian-roulette-style acceptance of sample.
        if(rng.next1d() < step_density / sbt_data->density_majorant)
        {
            // Scattering or absorbtion event case.
            atcg::SampledSpectrum albedo =
                atcg::SampledSpectrum::fromRGB(sbt_data->albedo_grid.eval(step_position), wavelengths);
            atcg::SampledSpectrum emission =
                atcg::SampledSpectrum::fromRGB(sbt_data->emission_grid.eval(step_position), wavelengths);
            result.transmittance_weight = albedo;
            result.emission_weight      = emission;
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

extern "C" __device__ float __direct_callable__heterogeneousMedium_evalTransmittance(const glm::vec3& origin,
                                                                                     const glm::vec3& direction,
                                                                                     float distance,
                                                                                     atcg::PCG32& rng)
{
    const atcg::HeterogeneousMediumData* sbt_data =
        *reinterpret_cast<const atcg::HeterogeneousMediumData**>(optixGetSbtDataPointer());
    atcg::TransmittanceEstimator<atcg::TransmittanceSamplingStrategyType::RATIO_TRACKING,
                                 decltype(sbt_data->density_grid)>
        estimator(sbt_data->density_majorant, sbt_data->density_grid);
    atcg::Ray ray(origin, direction, 0.0f, distance);
    return estimator.estimate(ray, rng);
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
    result.interaction.position           = origin + sample.distance * direction;
    result.transmittance_weight           = sample.transmittance_weight;
    result.radiance_weight                = sample.emission_weight;
    if(result.interaction.isValid())
    {
        result.interaction.position = origin + result.interaction.incoming_distance * direction;
    }
    return result;
}