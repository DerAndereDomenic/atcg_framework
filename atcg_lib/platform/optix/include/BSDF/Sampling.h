#pragma once

#include <BSDF/BSDFFunctions.h>

namespace atcg
{
/**
 * @brief Sample a direction according to the GGX normal distribution function
 *
 * @param uv The random numbers used for sampling
 * @param roughness The surface roughness
 *
 * @return The sampled direction
 */
template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto warp_square_to_hemisphere_ggx(const glm::vec2& uv, const T& roughness)
{
    // GGX NDF sampling
    auto cos_theta = CuDiff::sqrt(CuDiff::max(1e-12f, (1.0f - uv.x) / (1.0f + (roughness * roughness - 1.0f) * uv.x)));
    auto sin_theta = CuDiff::sqrt(CuDiff::max(1e-12f, 1.0f - cos_theta * cos_theta));
    float phi      = 2.0f * glm::pi<float>() * uv.y;

    auto x = sin_theta * glm::cos(phi);
    auto y = sin_theta * glm::sin(phi);
    auto z = cos_theta;

    auto res = CuDiff::wrap(x, y, z);

    return res;
}

/**
 * @brief Evaluate the pdf of sampling a direction according to the GGX normal distribution function
 *
 * @param result The direction
 * @param roughness The surface roughness
 *
 * @return The pdf result
 */
template<typename resultType, typename roughnessType>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto warp_square_to_hemisphere_ggx_pdf(const resultType& result,
                                                                          roughnessType roughness)
{
    auto [rx, ry, rz] = CuDiff::unwrap(result);
    return D_GGX(rz, roughness) * CuDiff::max(0.0f, rz);
}

/**
 * @brief Sample a direction in the hemisphere using cosine weighted sampling
 *
 * @param uv The random numbers used for sampling
 *
 * @return The direction
 */
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
 * @brief Jacobian of transforming a halfway direction to reflected direction
 *
 * @param HdotV Dot product between halfway and incoming direction
 * @param HdotL Dot product between halfway and outgoing direction
 * @param eta Index of refraction n1/n2
 *
 * @return The pdf
 */
template<typename HdotVType, typename HdotLType, typename etaType>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto
warp_normal_to_refracted_direction_pdf(const HdotVType HdotV, const HdotLType HdotL, const etaType eta)
{
    auto denom = (HdotL + eta * HdotV);
    return eta * eta * CuDiff::abs(HdotV) / (denom * denom);
}

/**
 * @brief Jacobian of transforming a halfway direction to reflected direction
 *
 * @param reflected_dir The reflected direction
 * @param normal The surface normal
 *
 * @return The pdf
 */
template<typename T, typename U>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto warp_normal_to_reflected_direction_pdf(const T& reflected_dir, const U& normal)
{
    return 1.0f / CuDiff::abs(4.0f * CuDiff::dot(reflected_dir, normal));
}
}    // namespace atcg