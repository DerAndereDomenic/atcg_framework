#pragma once

#include <BSDF/BSDFFunctions.h>

namespace atcg
{

enum class SamplingStrategyType
{
    HEMISPHERE_GGX,
    HEMISPHERE_COSINE,
    SPHERE_UNIFORM,
    EXPONENTIAL_SAMPLING
};

template<SamplingStrategyType strategy>
struct SamplingStrategy;

template<>
struct SamplingStrategy<SamplingStrategyType::HEMISPHERE_GGX>
{
    float _roughness;
    ATCG_HOST_DEVICE ATCG_FORCE_INLINE SamplingStrategy<SamplingStrategyType::HEMISPHERE_GGX>(float roughness)
        : _roughness(roughness)
    {
    }

    ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec3 sample(const glm::vec2& uv)
    {
        // GGX NDF sampling
        float cos_theta = glm::sqrt((1.0f - uv.x) / (1.0f + (_roughness * _roughness - 1.0f) * uv.x));
        float sin_theta = glm::sqrt(glm::max(0.0f, 1.0f - cos_theta * cos_theta));
        float phi       = 2.0f * glm::pi<float>() * uv.y;

        float x = sin_theta * glm::cos(phi);
        float y = sin_theta * glm::sin(phi);
        float z = cos_theta;

        return glm::vec3(x, y, z);
    }

    ATCG_HOST_DEVICE ATCG_FORCE_INLINE float pdf(const glm::vec3& result)
    {
        return D_GGX(result.z, _roughness) * glm::max(0.0f, result.z);
    }

    static ATCG_HOST_DEVICE ATCG_FORCE_INLINE float
    warp_halfway_to_reflected_direction_pdf(const glm::vec3& reflected_dir, const glm::vec3& normal)
    {
        return 1 / glm::abs(4 * glm::dot(reflected_dir, normal));
    }

    static ATCG_HOST_DEVICE ATCG_FORCE_INLINE float
    warp_halfway_to_refracted_direction_pdf(const float HdotV, const float HdotL, const float eta)
    {
        float denom = (HdotL + eta * HdotV);
        return eta * eta * glm::abs(HdotV) / (denom * denom);
    }
};

template<>
struct SamplingStrategy<SamplingStrategyType::HEMISPHERE_COSINE>
{
    ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec3 sample(const glm::vec2& uv)
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

    ATCG_HOST_DEVICE ATCG_FORCE_INLINE float pdf(const glm::vec3& result)
    {
        return glm::max(0.0f, result.z) / glm::pi<float>();
    }
};

template<>
struct SamplingStrategy<SamplingStrategyType::SPHERE_UNIFORM>
{
    ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec3 sample(const glm::vec2& uv)
    {
        float z   = 1.0f - 2.0f * uv.x;
        float r   = glm::sqrt(glm::max(0.0f, 1.0f - z * z));
        float phi = 2.0f * glm::pi<float>() * uv.y;

        float x = r * glm::cos(phi);
        float y = r * glm::sin(phi);

        return glm::vec3(x, y, z);
    }

    ATCG_HOST_DEVICE ATCG_FORCE_INLINE float pdf(const glm::vec3& /*result*/)
    {
        return 1.0f / (4.0f * glm::pi<float>());
    }
};
}    // namespace atcg