#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>
#include <Utils/HostDevice.h>
#include <Core/SurfaceInteraction.h>

#include <Emitter/EmitterVPtrTable.cuh>
#include <Emitter/MeshEmitterData.cuh>
#include <DataStructure/Frame.h>
#include <BSDF/Sampling.h>

namespace detail
{
/**
 * @brief Eval a mesh emitter
 *
 * @param emissive_color The emissive color
 * @param emitter_scaling The scaling (intensity) of the emitter
 *
 * @return The radiance value
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec3 evalMeshEmitter(const glm::vec3& emissive_color,
                                                             const float emitter_scaling)
{
    return emissive_color * emitter_scaling;
}

/**
 * @brief Sample a mesh emitter
 *
 * @param si The surface interaction to sample from
 * @param sampler The shape sampler
 * @param rng The rng
 *
 * @return The sampling result
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE atcg::EmitterSamplingResult
sampleMeshEmitter(const atcg::AnyInteraction& si, const atcg::ShapeSamplerVPtrTable* sampler, atcg::PCG32& rng)
{
    atcg::EmitterSamplingResult result;

    atcg::ShapeSampleResult shape_sample_result = sampler->sampleShape(rng);

    glm::vec3 light_position = shape_sample_result.position;
    glm::vec3 light_normal   = shape_sample_result.normal;
    float total_area         = 1.0f / shape_sample_result.pdf_dA;

    // Assemble sampling result
    result.pdf_dw = 0;    // initialize with invalid sample

    // light source sampling
    result.direction_to_light       = glm::normalize(light_position - si->position);
    float distance_to_light_squared = glm::length2(light_position - si->position) + 1e-5f;
    result.distance_to_light        = glm::length(light_position - si->position) + 1e-5f;
    result.normal_at_light          = light_normal;
    result.uvs                      = shape_sample_result.uvs;

    float one_over_light_position_pdf  = total_area;
    float cos_theta_on_light           = glm::abs(glm::dot(result.direction_to_light, light_normal));
    float one_over_light_direction_pdf = one_over_light_position_pdf * cos_theta_on_light / distance_to_light_squared;


    // Probability of sampling this direction via light source sampling
    result.pdf_dw = 1 / (one_over_light_direction_pdf + 1e-5f);

    return result;
}

/**
 * @brief Evaluate the pdf of a mesh emitter
 *
 * @param last_si The last surface interaction
 * @param sampler The shape sampler
 * @param si The current surface interaction
 *
 * @return The pdf
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE float evalMeshEmitterPDF(const atcg::AnyInteraction& last_si,
                                                            const atcg::ShapeSamplerVPtrTable* sampler,
                                                            const atcg::SurfaceInteraction& si)
{
    // We can assume that outgoing ray dir actually intersects the light source.

    // Some useful quantities
    glm::vec3 light_normal         = si.normal;
    glm::vec3 light_ray_dir        = glm::normalize(si.position - last_si->position);
    float light_ray_length_squared = glm::length2(si.position - last_si->position);

    // The probability of sampling any position on the surface of the mesh is the reciprocal of its surface area.
    float light_position_pdf = sampler->evalShapePdf();

    // Probability of sampling this direction via light source sampling
    float cos_theta_on_light  = glm::abs(glm::dot(light_ray_dir, light_normal));
    float light_direction_pdf = light_position_pdf * light_ray_length_squared / cos_theta_on_light;

    return light_direction_pdf;
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE atcg::PhotonSamplingResult
samplePhoton(const atcg::MeshEmitterData* sbt_data, const atcg::SampledWavelengths& wavelengths, atcg::PCG32& rng)
{
    atcg::PhotonSamplingResult result;

    atcg::ShapeSampleResult shape_sample_result = sbt_data->sampler->sampleShape(rng);

    glm::vec3 light_normal   = shape_sample_result.normal;
    glm::vec3 light_position = shape_sample_result.position;

    atcg::Frame<glm::vec3> frame(light_normal);

    atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_COSINE> sampling_strategy;
    glm::vec3 local_dir = sampling_strategy.sample(rng.next2d());
    glm::vec3 world_dir = frame.toWorld(local_dir);

    float position_pdf  = sbt_data->sampler->evalShapePdf();
    float direction_pdf = sampling_strategy.pdf(local_dir);
    float pdf           = position_pdf * direction_pdf;

    result.position        = light_position;
    result.direction       = world_dir;
    result.normal          = light_normal;
    result.pdf_dA_dw       = pdf;
    result.radiance_weight = atcg::SampledSpectrum::fromRGB(sbt_data->emitter_scaling *
                                                                sbt_data->emissive_texture.read(glm::vec2(result.uvs)),
                                                            wavelengths) *
                             glm::pi<float>() / position_pdf;
    result.uvs             = shape_sample_result.uvs;

    return result;
}

}    // namespace detail

extern "C" __device__ atcg::EmitterSamplingResult
__direct_callable__sample_meshemitter(const atcg::AnyInteraction& si,
                                      const atcg::SampledWavelengths& wavelengths,
                                      atcg::PCG32& rng)
{
    const atcg::MeshEmitterData* sbt_data = *reinterpret_cast<const atcg::MeshEmitterData**>(optixGetSbtDataPointer());
    atcg::EmitterSamplingResult result    = detail::sampleMeshEmitter(si, sbt_data->sampler, rng);

    glm::vec3 emissive_color = sbt_data->emissive_texture.read(glm::vec2(result.uvs));

    result.radiance_weight_at_receiver =
        atcg::SampledSpectrum::fromRGB(sbt_data->emitter_scaling * emissive_color, wavelengths) / result.pdf_dw;

    return result;
}

extern "C" __device__ atcg::SampledSpectrum
__direct_callable__eval_meshemitter(const atcg::SurfaceInteraction& si, const atcg::SampledWavelengths& wavelengths)
{
    const atcg::MeshEmitterData* sbt_data = *reinterpret_cast<const atcg::MeshEmitterData**>(optixGetSbtDataPointer());

    glm::vec3 emissive_color = sbt_data->emissive_texture.read(si.uv);

    return atcg::SampledSpectrum::fromRGB(sbt_data->emitter_scaling * emissive_color, wavelengths);
}

extern "C" __device__ float __direct_callable__evalpdf_meshemitter(const atcg::AnyInteraction& last_si,
                                                                   const atcg::SurfaceInteraction& si)
{
    const atcg::MeshEmitterData* sbt_data = *reinterpret_cast<const atcg::MeshEmitterData**>(optixGetSbtDataPointer());
    // We can assume that outgoing ray dir actually intersects the light source.

    return detail::evalMeshEmitterPDF(last_si, sbt_data->sampler, si);
}

extern "C" __device__ atcg::PhotonSamplingResult
__direct_callable__samplephoton_meshemitter(const atcg::SampledWavelengths& wavelengths, atcg::PCG32& rng)
{
    const atcg::MeshEmitterData* sbt_data = *reinterpret_cast<const atcg::MeshEmitterData**>(optixGetSbtDataPointer());
    return detail::samplePhoton(sbt_data, wavelengths, rng);
}
