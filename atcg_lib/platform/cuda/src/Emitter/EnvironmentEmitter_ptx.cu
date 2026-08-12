#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>
#include <Utils/HostDevice.h>
#include <DataStructure/SurfaceInteraction.h>

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

    float phi   = std::atan2(ray_dir.z, ray_dir.x);
    float theta = std::acos(ray_dir.y);

    float u = (phi + glm::pi<float>()) / (2.0f * glm::pi<float>());
    float v = theta / glm::pi<float>();

    glm::vec2 uv(u, v);

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
ATCG_HOST_DEVICE ATCG_FORCE_INLINE atcg::EmitterSamplingResult
sampleEnvironmentEmitter(const atcg::EnvironmentEmitterData* sbt_data, const atcg::AnyInteraction& ai, atcg::PCG32& rng)
{
    atcg::EmitterSamplingResult result;

    // Sample pixel
    uint32_t row_index = atcg::Math::binary_search(sbt_data->row_cdf, rng.nextFloat(), sbt_data->height);
    uint32_t col_index =
        atcg::Math::binary_search(sbt_data->col_cdfs + row_index * sbt_data->width, rng.nextFloat(), sbt_data->width);

    row_index = glm::clamp(row_index, 0u, static_cast<uint32_t>(sbt_data->height - 1));
    col_index = glm::clamp(col_index, 0u, static_cast<uint32_t>(sbt_data->width - 1));

    glm::vec2 jitter = rng.next2d();
    float u          = (col_index + jitter.x) / sbt_data->width;
    float v          = (row_index + jitter.y) / sbt_data->height;

    float phi   = u * 2.0f * glm::pi<float>() - glm::pi<float>();
    float theta = v * glm::pi<float>();

    float pixel_pdf = sbt_data->row_pdf[row_index] * sbt_data->col_pdfs[row_index * sbt_data->width + col_index];
    float jacobian =
        (2.0f * glm::pi<float>() * glm::pi<float>() * std::sin(theta)) / float(sbt_data->width * sbt_data->height);
    float direction_pdf = pixel_pdf / jacobian;

    result.distance_to_light = std::numeric_limits<float>::infinity();
    result.uvs               = glm::vec3(u, v, 0);
    result.sampling_pdf      = direction_pdf;
    result.direction_to_light =
        glm::vec3(std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi));

    if(ai.is_surface())
    {
        if(glm::dot(result.direction_to_light, ai.si.normal) < 0)
        {
            result.sampling_pdf = 0.0f;    // Invalid
        }
    }

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
ATCG_HOST_DEVICE ATCG_FORCE_INLINE float evalEnvironmentEmitterSamplingPdf(const atcg::EnvironmentEmitterData* sbt_data,
                                                                           const atcg::AnyInteraction& last_si,
                                                                           const atcg::SurfaceInteraction& si)
{
    // We can assume that outgoing ray dir actually intersects the light source.

    float phi   = std::atan2(si.incoming_direction.z, si.incoming_direction.x);
    float theta = std::acos(si.incoming_direction.y);

    float u = (phi + glm::pi<float>()) * glm::one_over_two_pi<float>();
    float v = theta / glm::pi<float>();

    uint32_t col_index =
        glm::clamp(static_cast<uint32_t>(u * sbt_data->width), 0u, static_cast<uint32_t>(sbt_data->width - 1));
    uint32_t row_index =
        glm::clamp(static_cast<uint32_t>(v * sbt_data->height), 0u, static_cast<uint32_t>(sbt_data->height - 1));

    float pixel_pdf = sbt_data->row_pdf[row_index] * sbt_data->col_pdfs[row_index * sbt_data->width + col_index];
    float jacobian =
        (2.0f * glm::pi<float>() * glm::pi<float>() * std::sin(theta)) / float(sbt_data->width * sbt_data->height);
    float direction_pdf = pixel_pdf / jacobian;

    return direction_pdf;
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec2 concentricSampleDisk(const glm::vec2& u)
{
    // Map [0,1)^2 to [-1,1]^2
    glm::vec2 uOffset = 2.0f * u - glm::vec2(1, 1);

    if(uOffset.x == 0 && uOffset.y == 0) return glm::vec2(0, 0);

    float theta, r;
    if(std::abs(uOffset.x) > std::abs(uOffset.y))
    {
        r     = uOffset.x;
        theta = glm::quarter_pi<float>() * (uOffset.y / uOffset.x);
    }
    else
    {
        r     = uOffset.y;
        theta = glm::half_pi<float>() - glm::quarter_pi<float>() * (uOffset.x / uOffset.y);
    }

    return r * glm::vec2(std::cos(theta), std::sin(theta));
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE atcg::PhotonSamplingResult samplePhoton(const atcg::EnvironmentEmitterData* sbt_data,
                                                                           const atcg::SampledWavelengths& wavelengths,
                                                                           atcg::PCG32& rng)
{
    atcg::PhotonSamplingResult result;

    // --- Step 1: importance-sample a direction from the environment map ---
    // (identical to the direction-sampling part of sampleEnvironmentEmitter)
    uint32_t row_index = atcg::Math::binary_search(sbt_data->row_cdf, rng.nextFloat(), sbt_data->height);
    uint32_t col_index =
        atcg::Math::binary_search(sbt_data->col_cdfs + row_index * sbt_data->width, rng.nextFloat(), sbt_data->width);

    row_index = glm::clamp(row_index, 0u, static_cast<uint32_t>(sbt_data->height - 1));
    col_index = glm::clamp(col_index, 0u, static_cast<uint32_t>(sbt_data->width - 1));

    glm::vec2 jitter = rng.next2d();
    float u          = (col_index + jitter.x) / sbt_data->width;
    float v          = (row_index + jitter.y) / sbt_data->height;

    float phi   = u * 2.0f * glm::pi<float>() - glm::pi<float>();
    float theta = v * glm::pi<float>();

    float pixel_pdf = sbt_data->row_pdf[row_index] * sbt_data->col_pdfs[row_index * sbt_data->width + col_index];
    float jacobian =
        (2.0f * glm::pi<float>() * glm::pi<float>() * std::sin(theta)) / float(sbt_data->width * sbt_data->height);
    float direction_pdf = pixel_pdf / jacobian;

    // Direction pointing FROM the scene OUT towards the environment (same convention as NEE sampling)
    glm::vec3 direction_to_light =
        glm::vec3(std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi));

    // The photon travels the opposite way: from the environment INTO the scene
    glm::vec3 photon_direction = -direction_to_light;

    // --- Step 2: sample a position on a disk perpendicular to the direction, ---
    // --- at the scene's bounding sphere                                       ---
    atcg::Frame<glm::vec3> disk_frame(direction_to_light);

    glm::vec3 scene_center = 0.5f * (sbt_data->bounding_box.min + sbt_data->bounding_box.max);
    float scene_radius     = 0.5f * glm::length(sbt_data->bounding_box.max - sbt_data->bounding_box.min) *
                             1.01f;    // Slightly enlarge to avoid numerical issues

    glm::vec2 disk_sample   = concentricSampleDisk(rng.next2d()) * scene_radius;
    glm::vec3 disk_position = scene_center + direction_to_light * scene_radius +
                              disk_frame.toWorld(glm::vec3(disk_sample.x, disk_sample.y, 0.0f));

    float position_pdf = 1.0f / (glm::pi<float>() * scene_radius * scene_radius);
    float pdf          = position_pdf * direction_pdf;

    // --- Step 3: look up radiance and assemble the result ---
    glm::vec3 color = sbt_data->environment_texture.read(glm::vec2(u, 1.0f - v));

    result.position  = disk_position;
    result.direction = photon_direction;
    result.normal    = -direction_to_light;    // disk faces into the scene
    result.uvs       = glm::vec3(u, v, 0);
    result.pdf       = pdf;
    // No cosine term here (unlike the Lambertian mesh case): the disk is constructed
    // to be perpendicular to the propagation direction by definition, so cos = 1.
    result.radiance_weight = atcg::SampledSpectrum::fromRGB(color, wavelengths) / pdf;

    return result;
}

}    // namespace detail

extern "C" __device__ atcg::EmitterSamplingResult
__direct_callable__sample_environmentemitter(const atcg::AnyInteraction& si,
                                             const atcg::SampledWavelengths& wavelengths,
                                             atcg::PCG32& rng)
{
    const atcg::EnvironmentEmitterData* sbt_data =
        *reinterpret_cast<const atcg::EnvironmentEmitterData**>(optixGetSbtDataPointer());

    atcg::EmitterSamplingResult result = detail::sampleEnvironmentEmitter(sbt_data, si, rng);

    glm::vec3 color = sbt_data->environment_texture.read(glm::vec2(result.uvs.x, 1.0f - result.uvs.y));

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

    return detail::evalEnvironmentEmitterSamplingPdf(sbt_data, last_si, si);
}

extern "C" __device__ atcg::PhotonSamplingResult
__direct_callable__samplephoton_environmentemitter(const atcg::SampledWavelengths& wavelengths, atcg::PCG32& rng)
{
    const atcg::EnvironmentEmitterData* sbt_data =
        *reinterpret_cast<const atcg::EnvironmentEmitterData**>(optixGetSbtDataPointer());

    return detail::samplePhoton(sbt_data, wavelengths, rng);
}