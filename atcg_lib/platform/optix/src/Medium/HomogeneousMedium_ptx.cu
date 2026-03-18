#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>

#include <Medium/HomogeneousMediumData.cuh>
#include <Medium/MediumVPtrTable.cuh>
#include <Core/GlobalAtomicAdd.h>
#include <Medium/Transmittance.h>


extern "C" __device__ float __direct_callable__homogeneousMedium_evalTransmittance(const glm::vec3& origin,
                                                                                   const glm::vec3& direction,
                                                                                   float distance,
                                                                                   atcg::PCG32& unused_rng)
{
    const atcg::HomogeneousMediumData* sbt_data =
        *reinterpret_cast<const atcg::HomogeneousMediumData**>(optixGetSbtDataPointer());

    // Evaluate the probability of the light *not* interacting with the medium.
    float density = *(sbt_data->density);

    atcg::TransmittanceEstimator<atcg::TransmittanceSamplingStrategyType::HOMOGENEOUS_TRACKING> estimator(density);
    atcg::Ray ray(origin, direction, 0.0f, distance);
    return estimator.estimate(ray);
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
    glm::vec3 albedo_            = *(sbt_data->albedo);
    atcg::SampledSpectrum albedo = atcg::SampledSpectrum::fromRGB(albedo_, wavelengths);
    atcg::SampledSpectrum Le     = atcg::SampledSpectrum::fromRGB(sbt_data->Le, wavelengths);

    // Scalar projection of scattering coefficient, used to sample the next medium scattering event.
    float sigma_t_scalar = *(sbt_data->density);
    glm::vec3 sigma_s    = albedo_ * sigma_t_scalar;

    atcg::MediumSamplingResult result;
    // Dummy implementation:
    // Effectively no medium event.
    result.interaction                    = atcg::MediumInteraction();
    result.interaction.incoming_direction = direction;
    result.transmittance_weight           = atcg::SampledSpectrum(1);
    result.radiance_weight                = atcg::SampledSpectrum(0);

    // Sample the free-flight distance proportional to sigma_s_scalar.
    atcg::SamplingStrategy<atcg::SamplingStrategyType::EXPONENTIAL_SAMPLING> sampling_strategy(sigma_t_scalar);
    float sampled_distance = sampling_strategy.sample(rng.next1d());

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
        // result.transmittance_value =
        //     atcg::SampledSpectrum(sigma_s * detail::transmittance(sampled_distance, sigma_t_scalar));
        // result.transmittance_pdf =
        //     detail::warp_1d_sample_to_medium_event_distance_pdf(sampled_distance, sigma_t_scalar);

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
        // result.transmittance_value  = atcg::SampledSpectrum(detail::transmittance(max_distance, sigma_t_scalar));
        // result.transmittance_pdf    = detail::transmittance(max_distance, sigma_t_scalar);
        // detail::transmittance(max_distance, sigma_t - atcg::SampledSpectrum(sigma_t_scalar));
    }

    return result;
}

extern "C" __device__ void
__direct_callable__homogeneousMedium_sampleMediumEventBackward(const glm::vec3& origin,
                                                               const glm::vec3& direction,
                                                               float max_distance,
                                                               const atcg::SampledWavelengths& wavelengths,
                                                               atcg::PCG32& rng,
                                                               const glm::vec3& output_grad)
{
    const atcg::HomogeneousMediumData* sbt_data =
        *reinterpret_cast<const atcg::HomogeneousMediumData**>(optixGetSbtDataPointer());

    if(!sbt_data->optimize_albedo && !sbt_data->optimize_density)
    {
        // Nothing to do
        return;
    }

    // Absorbtion, scattering and extinction coefficients...
    glm::vec3 albedo_            = *(sbt_data->albedo);
    atcg::SampledSpectrum albedo = atcg::SampledSpectrum::fromRGB(albedo_, wavelengths);

    // Scalar projection of scattering coefficient, used to sample the next medium scattering event.
    float sigma_t_scalar = *(sbt_data->density);
    glm::vec3 sigma_s    = albedo_ * sigma_t_scalar;


    // Sample the free-flight distance proportional to sigma_s_scalar.
    atcg::SamplingStrategy<atcg::SamplingStrategyType::EXPONENTIAL_SAMPLING> sampling_strategy(sigma_t_scalar);
    float sampled_distance = sampling_strategy.sample(rng.next1d());

    if(sampled_distance < max_distance)
    {
        // Medium event!

        atcg::TransmittanceEstimator<atcg::TransmittanceSamplingStrategyType::HOMOGENEOUS_TRACKING> estimator(
            sigma_t_scalar);
        float T = estimator.estimate(atcg::Ray(origin, direction, 0.0f, sampled_distance));

        // sigma_t_scalar * T * output_grad / sigma_s * T
        glm::vec3 albedo_gradient = sigma_t_scalar / sigma_s * output_grad;    // / (sigma_s * T);
        // glm::dot((1.0f - sampled_distance * sigma_t_scalar) * albedo_ * T, output_grad) / (sigma_s * T)
        float density_gradient =
            glm::dot(glm::vec3((1.0f - sampled_distance * sigma_t_scalar) / sigma_t_scalar), output_grad);

        if(sbt_data->optimize_albedo)
        {
            if(isfinite(albedo_gradient.x) && isfinite(albedo_gradient.y) && isfinite(albedo_gradient.z))
            {
                atcg::globalAtomicAdd(sbt_data->albedo_grad + 0, albedo_gradient.x);
                atcg::globalAtomicAdd(sbt_data->albedo_grad + 1, albedo_gradient.y);
                atcg::globalAtomicAdd(sbt_data->albedo_grad + 2, albedo_gradient.z);
            }
        }

        if(sbt_data->optimize_density)
        {
            if(isfinite(density_gradient))
            {
                atcg::globalAtomicAdd(sbt_data->density_grad, density_gradient);
            }
        }
    }
    else
    {
        // No emission, no absorption, no scattering
        // No medium event...
        // The sampling did not succeed, and there is no scattering event *before* the max_distance.
        // This is independent of the volume albedo and density, so no gradients to those parameters.
        if(sbt_data->optimize_density)
        {
            // glm::dot(glm::vec3(-max_distance * detail::transmittance(max_distance, sigma_t_scalar)), output_grad) / T
            float density_gradient = glm::dot(glm::vec3(-max_distance), output_grad);

            atcg::globalAtomicAdd(sbt_data->density_grad, density_gradient);
        }
    }
}