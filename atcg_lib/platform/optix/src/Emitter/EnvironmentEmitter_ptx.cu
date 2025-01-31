#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>
#include <Math/Functions.h>
#include <Core/SurfaceInteraction.h>

#include <Emitter/EmitterVPtrTable.cuh>
#include <Emitter/EnvironmentEmitterData.cuh>

namespace detail
{

ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec3 warp_square_to_hemisphere_cosine(const glm::vec2& uv)
{
    // Sample disk uniformly
    float r   = glm::sqrt(uv.x);
    float phi = 2.0f * glm::pi<float>() * uv.y;

    // Project disk sample onto hemisphere
    float x = r * glm::cos(phi);
    float y = r * glm::sin(phi);
    float z = glm::sqrt(glm::max(0.0f, 1 - uv.x));

    return glm::vec3(x, y, z);
}

/**
 * @brief Evaluate the pdf of sampling a direction according to a cosine weighted distribution
 *
 * @param result The direction
 *
 * @return The pdf result
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE float warp_square_to_hemisphere_cosine_pdf(const glm::vec3& result)
{
    return glm::max(0.0f, result.z) / glm::pi<float>();
}

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
ATCG_HOST_DEVICE ATCG_FORCE_INLINE atcg::EmitterSamplingResult
sampleEnvironmentEmitter(const atcg::SurfaceInteraction& si, atcg::PCG32& rng)
{
    atcg::EmitterSamplingResult result;

    glm::vec3 random_dir = warp_square_to_hemisphere_cosine(rng.next2d());
    float pdf            = warp_square_to_hemisphere_cosine_pdf(random_dir);
    glm::mat3 frame      = atcg::Math::compute_local_frame(si.normal);

    random_dir = frame * random_dir;

    glm::vec3 ray_dir = si.incoming_direction;

    float theta = std::acos(ray_dir.y) / glm::pi<float>();
    float phi   = (std::atan2(ray_dir.z, ray_dir.x) + glm::pi<float>()) / (2.0f * glm::pi<float>());

    glm::vec3 uv(phi, theta, 0);

    result.distance_to_light = std::numeric_limits<float>::infinity();
    result.sampling_pdf      = pdf;
    result.uvs               = uv;

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
ATCG_HOST_DEVICE ATCG_FORCE_INLINE float evalEnvironmentEmitterSamplingPdf(const atcg::SurfaceInteraction& last_si,
                                                                           const atcg::SurfaceInteraction& si)
{
    // We can assume that outgoing ray dir actually intersects the light source.

    // Probability of sampling this direction via light source sampling
    glm::mat3 local_frame     = atcg::Math::compute_local_frame(last_si.normal);
    glm::vec3 local_direction = glm::transpose(local_frame) * si.incoming_direction;

    return warp_square_to_hemisphere_cosine_pdf(local_direction);
}
}    // namespace detail

extern "C" __device__ atcg::EmitterSamplingResult
__direct_callable__sample_environmentemitter(const atcg::SurfaceInteraction& si, atcg::PCG32& rng)
{
    const atcg::EnvironmentEmitterData* sbt_data =
        *reinterpret_cast<const atcg::EnvironmentEmitterData**>(optixGetSbtDataPointer());

    atcg::EmitterSamplingResult result = detail::sampleEnvironmentEmitter(si, rng);

    float4 color = tex2D<float4>(sbt_data->environment_texture, result.uvs.x, 1.0f - result.uvs.y);

    result.distance_to_light           = std::numeric_limits<float>::infinity();
    result.sampling_pdf                = result.sampling_pdf;
    result.radiance_weight_at_receiver = glm::vec3(color.x, color.y, color.z) / result.sampling_pdf;

    return result;
}

extern "C" __device__ glm::vec3 __direct_callable__eval_environmentemitter(const atcg::SurfaceInteraction& si)
{
    const atcg::EnvironmentEmitterData* sbt_data =
        *reinterpret_cast<const atcg::EnvironmentEmitterData**>(optixGetSbtDataPointer());

    glm::vec2 uv = detail::evalEnvironmentEmitter(si);

    float4 color = tex2D<float4>(sbt_data->environment_texture, uv.x, 1.0f - uv.y);

    return glm::vec3(color.x, color.y, color.z);
}

extern "C" __device__ float __direct_callable__evalpdf_environmentemitter(const atcg::SurfaceInteraction& last_si,
                                                                          const atcg::SurfaceInteraction& si)
{
    const atcg::EnvironmentEmitterData* sbt_data =
        *reinterpret_cast<const atcg::EnvironmentEmitterData**>(optixGetSbtDataPointer());
    // We can assume that outgoing ray dir actually intersects the light source.

    return detail::evalEnvironmentEmitterSamplingPdf(last_si, si);
}