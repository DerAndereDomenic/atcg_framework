#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>
#include <Math/Functions.h>
#include <Core/SurfaceInteraction.h>

#include <Emitter/EmitterVPtrTable.cuh>
#include <Emitter/EnvironmentEmitterData.cuh>

#include <DataStructure/Frame.h>
#include <BSDF/Sampling.h>

namespace detail
{

/**
 * @brief Evaluate an environment emitter
 *
 * @param si The surface interaction
 *
 * @return The uv cooridnates to perform the texture lookup
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec2 evalEnvironmentEmitter(const atcg::SurfaceInteraction& si)
{
    glm::vec3 ray_dir = si.incoming_direction;

    float theta = std::acos(ray_dir.y) / glm::pi<float>();
    float phi   = (std::atan2(ray_dir.z, ray_dir.x) + glm::pi<float>()) / (2.0f * glm::pi<float>());

    glm::vec2 uv(phi, theta);

    return uv;
}

/**
 * @brief Sample an environment emitter
 *
 * @param si The surface interaction
 * @param rng The rng
 *
 * @return The sampling result
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE atcg::EmitterSamplingResult sampleEnvironmentEmitter(const atcg::AnyInteraction& ai,
                                                                                        atcg::PCG32& rng)
{
    atcg::EmitterSamplingResult result;

    glm::vec3 random_dir;
    float pdf;
    if(ai.is_surface())
    {
        atcg::SurfaceInteraction si = ai;
        atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_COSINE> strategy;
        random_dir        = strategy.sample(rng.next2d());
        pdf               = strategy.pdf(random_dir);
        atcg::Frame frame = atcg::Frame(si.normal);

        random_dir = frame.toWorld(random_dir);
    }
    else if(ai.is_medium())
    {
        atcg::SamplingStrategy<atcg::SamplingStrategyType::SPHERE_UNIFORM> strategy;
        random_dir = strategy.sample(rng.next2d());
        pdf        = strategy.pdf(random_dir);
    }
    else
    {
        printf("Evaluated environment emitter with invalid interaction type. This should not happen.\n");
        return result;
    }


    glm::vec3 ray_dir = random_dir;

    float theta = std::acos(ray_dir.y) / glm::pi<float>();
    float phi   = (std::atan2(ray_dir.z, ray_dir.x) + glm::pi<float>()) / (2.0f * glm::pi<float>());

    glm::vec3 uv(phi, theta, 0);

    result.distance_to_light  = std::numeric_limits<float>::infinity();
    result.sampling_pdf       = pdf;
    result.uvs                = uv;
    result.direction_to_light = random_dir;

    return result;
}

/**
 * @brief Evaluate the pdf of an environment emitter
 *
 * @param last_si The last surface interaction
 * @param si The current surface interaction
 *
 * @return The pdf
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE float evalEnvironmentEmitterSamplingPdf(const atcg::AnyInteraction& last_si,
                                                                           const atcg::SurfaceInteraction& si)
{
    // We can assume that outgoing ray dir actually intersects the light source.

    // Probability of sampling this direction via light source sampling
    if(last_si.is_surface())
    {
        atcg::Frame frame            = atcg::Frame(last_si.si.normal);
        glm::vec3 local_dir_to_light = frame.toLocal(si.incoming_direction);
        atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_COSINE> strategy;
        return strategy.pdf(local_dir_to_light);
    }
    else if(last_si.is_medium())
    {
        atcg::SamplingStrategy<atcg::SamplingStrategyType::SPHERE_UNIFORM> strategy;
        return strategy.pdf(si.incoming_direction);
    }
    else
    {
        printf("Evaluated environment emitter sampling pdf with invalid interaction type. This should not happen.\n");
        return 0.0f;
    }
}
}    // namespace detail

extern "C" __device__ atcg::EmitterSamplingResult
__direct_callable__sample_environmentemitter(const atcg::AnyInteraction& si,
                                             const atcg::SampledWavelengths& wavelengths,
                                             atcg::PCG32& rng)
{
    const atcg::EnvironmentEmitterData* sbt_data =
        *reinterpret_cast<const atcg::EnvironmentEmitterData**>(optixGetSbtDataPointer());

    atcg::EmitterSamplingResult result = detail::sampleEnvironmentEmitter(si, rng);

    glm::vec3 color = sbt_data->environment_texture.read(glm::vec2(result.uvs.x, 1.0f - result.uvs.y));

    result.distance_to_light           = std::numeric_limits<float>::infinity();
    result.sampling_pdf                = result.sampling_pdf;
    result.radiance_weight_at_receiver = atcg::SampledSpectrum::fromRGB(color, wavelengths) / result.sampling_pdf;

    return result;
}

extern "C" __device__ atcg::SampledSpectrum
__direct_callable__eval_environmentemitter(const atcg::SurfaceInteraction& si,
                                           const atcg::SampledWavelengths& wavelengths)
{
    const atcg::EnvironmentEmitterData* sbt_data =
        *reinterpret_cast<const atcg::EnvironmentEmitterData**>(optixGetSbtDataPointer());

    glm::vec2 uv = detail::evalEnvironmentEmitter(si);

    return atcg::SampledSpectrum::fromRGB(sbt_data->environment_texture.read(glm::vec2(uv.x, 1.0f - uv.y)),
                                          wavelengths);
}

extern "C" __device__ float __direct_callable__evalpdf_environmentemitter(const atcg::AnyInteraction& last_si,
                                                                          const atcg::SurfaceInteraction& si)
{
    const atcg::EnvironmentEmitterData* sbt_data =
        *reinterpret_cast<const atcg::EnvironmentEmitterData**>(optixGetSbtDataPointer());
    // We can assume that outgoing ray dir actually intersects the light source.

    return detail::evalEnvironmentEmitterSamplingPdf(last_si, si);
}