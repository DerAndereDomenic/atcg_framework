#pragma once

#include <BSDF/BSDFFunctions.h>

namespace atcg
{

enum class SamplingStrategyType
{
    HEMISPHERE_GGX,
    HEMISPHERE_COSINE,
    SPHERE_UNIFORM,
    EXPONENTIAL_SAMPLING,
    HG_PHASE,
    RAYLEIGH_PHASE
};

template<SamplingStrategyType strategy, typename T = float>
struct SamplingStrategy;

template<typename T>
struct SamplingStrategy<SamplingStrategyType::HEMISPHERE_GGX, T>
{
    T _roughness;
    ATCG_HOST_DEVICE ATCG_FORCE_INLINE SamplingStrategy<SamplingStrategyType::HEMISPHERE_GGX, T>(T roughness)
        : _roughness(roughness)
    {
    }

    ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto sample(const glm::vec2& uv)
    {
        // GGX NDF sampling
        auto cos_theta = glm::sqrt((1.0f - uv.x) / (1.0f + (_roughness * _roughness - 1.0f) * uv.x));
        auto sin_theta = glm::sqrt(glm::max(0.0f, 1.0f - cos_theta * cos_theta));
        auto phi       = 2.0f * glm::pi<float>() * uv.y;

        auto x = sin_theta * glm::cos(phi);
        auto y = sin_theta * glm::sin(phi);
        auto z = cos_theta;

        return glm::vec3(x, y, z);
    }

    ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto pdf(const glm::vec3& result)
    {
        return D_GGX(result.z, _roughness) * glm::max(0.0f, result.z);
    }

    static ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto
    warp_halfway_to_reflected_direction_pdf(const glm::vec3& reflected_dir, const glm::vec3& normal)
    {
        return 1 / glm::abs(4 * glm::dot(reflected_dir, normal));
    }

    static ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto
    warp_halfway_to_refracted_direction_pdf(const float HdotV, const float HdotL, const float eta)
    {
        float denom = (HdotL + eta * HdotV);
        return eta * eta * glm::abs(HdotV) / (denom * denom);
    }
};

template<typename T>
struct SamplingStrategy<SamplingStrategyType::HEMISPHERE_COSINE, T>
{
    ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto sample(const glm::vec2& uv)
    {
        // Sample disk uniformly
        auto r   = glm::sqrt(uv.x);
        auto phi = 2.0f * glm::pi<float>() * uv.y;

        // Project disk sample onto hemisphere
        auto x = r * glm::cos(phi);
        auto y = r * glm::sin(phi);
        auto z = glm::sqrt(glm::max(0.0f, 1 - uv.x));

        return glm::vec3(x, y, z);
    }

    ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto pdf(const glm::vec3& result)
    {
        return glm::max(0.0f, result.z) / glm::pi<float>();
    }
};

template<typename T>
struct SamplingStrategy<SamplingStrategyType::SPHERE_UNIFORM, T>
{
    ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto sample(const glm::vec2& uv)
    {
        auto z   = 1.0f - 2.0f * uv.x;
        auto r   = glm::sqrt(glm::max(0.0f, 1.0f - z * z));
        auto phi = 2.0f * glm::pi<float>() * uv.y;

        auto x = r * glm::cos(phi);
        auto y = r * glm::sin(phi);

        return glm::vec3(x, y, z);
    }

    ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto pdf(const glm::vec3& /*result*/)
    {
        return 1.0f / (4.0f * glm::pi<float>());
    }
};
}    // namespace atcg