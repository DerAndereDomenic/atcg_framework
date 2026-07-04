#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>
#include <Utils/HostDevice.h>
#include <Core/SurfaceInteraction.h>

#include <Emitter/EmitterVPtrTable.cuh>
#include <Emitter/PointEmitterData.cuh>
#include <BSDF/Sampling.h>


extern "C" __device__ atcg::EmitterSamplingResult
__direct_callable__sample_pointemitter(const atcg::AnyInteraction& si,
                                       const atcg::SampledWavelengths& wavelengths,
                                       atcg::PCG32& rng)
{
    const atcg::PointEmitterData* sbt_data =
        *reinterpret_cast<const atcg::PointEmitterData**>(optixGetSbtDataPointer());

    atcg::EmitterSamplingResult result;

    glm::vec3 dir_to_light = sbt_data->position - si->position;
    float distance         = glm::length(dir_to_light);

    result.direction_to_light = dir_to_light / (1e-5f + distance);
    result.distance_to_light  = distance;
    result.radiance_weight_at_receiver =
        atcg::SampledSpectrum::fromRGB(sbt_data->color, wavelengths) * sbt_data->intensity / (distance * distance);
    result.sampling_pdf = 1.0f;

    return result;
}

extern "C" __device__ atcg::SampledSpectrum
__direct_callable__eval_pointemitter(const atcg::SurfaceInteraction& si, const atcg::SampledWavelengths& wavelengths)
{
    const atcg::PointEmitterData* sbt_data =
        *reinterpret_cast<const atcg::PointEmitterData**>(optixGetSbtDataPointer());

    return atcg::SampledSpectrum::fromRGB(sbt_data->color, wavelengths) * sbt_data->intensity;
}

extern "C" __device__ float __direct_callable__evalpdf_pointemitter(const atcg::AnyInteraction& last_si,
                                                                    const atcg::SurfaceInteraction& si)
{
    const atcg::PointEmitterData* sbt_data =
        *reinterpret_cast<const atcg::PointEmitterData**>(optixGetSbtDataPointer());

    return 0.0f;
}

extern "C" __device__ atcg::PhotonSamplingResult
__direct_callable__sample_photon_pointemitter(const atcg::SampledWavelengths& wavelengths, atcg::PCG32& rng)
{
    const atcg::PointEmitterData* sbt_data =
        *reinterpret_cast<const atcg::PointEmitterData**>(optixGetSbtDataPointer());

    atcg::PhotonSamplingResult result;

    atcg::SamplingStrategy<atcg::SamplingStrategyType::SPHERE_UNIFORM> sampling_strategy;

    result.position        = sbt_data->position;
    result.direction       = sampling_strategy.sample(rng.next2d());
    result.normal          = result.direction;
    result.radiance_weight = atcg::SampledSpectrum::fromRGB(sbt_data->color, wavelengths) * sbt_data->intensity;
    result.pdf             = sampling_strategy.pdf(result.direction);
    result.uvs             = glm::vec3(0.0f);

    return result;
}