#pragma once

#include <BSDF/BSDFFunctions.h>
#include <CuDiff/ext/glm.h>

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
        auto cos_theta = CuDiff::sqrt((1.0f - uv.x) / (1.0f + (_roughness * _roughness - 1.0f) * uv.x));
        auto sin_theta = CuDiff::sqrt(CuDiff::max(0.0f, 1.0f - cos_theta * cos_theta));
        auto phi       = 2.0f * glm::pi<float>() * uv.y;

        auto x = sin_theta * CuDiff::cos(phi);
        auto y = sin_theta * CuDiff::sin(phi);
        auto z = cos_theta;

        return CuDiff::wrap(x, y, z);
    }

    template<typename U>
    ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto pdf(const U& result)
    {
        auto [x, y, z] = CuDiff::unwrap(result);
        return D_GGX(z, _roughness) * CuDiff::max(0.0f, z);
    }

    template<typename U, typename V>
    static ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto warp_halfway_to_reflected_direction_pdf(const U& reflected_dir,
                                                                                           const V& normal)
    {
        return 1 / CuDiff::abs(4 * CuDiff::dot(reflected_dir, normal));
    }

    template<typename HdotVType, typename HdotLType, typename etaType>
    static ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto
    warp_halfway_to_refracted_direction_pdf(const HdotVType HdotV, const HdotLType HdotL, const etaType eta)
    {
        auto denom = (HdotL + eta * HdotV);
        return eta * eta * CuDiff::abs(HdotV) / (denom * denom);
    }
};

template<typename T>
struct SamplingStrategy<SamplingStrategyType::HEMISPHERE_COSINE, T>
{
    ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto sample(const glm::vec2& uv)
    {
        // Sample disk uniformly
        auto r   = CuDiff::sqrt(uv.x);
        auto phi = 2.0f * glm::pi<float>() * uv.y;

        // Project disk sample onto hemisphere
        auto x = r * CuDiff::cos(phi);
        auto y = r * CuDiff::sin(phi);
        auto z = CuDiff::sqrt(CuDiff::max(0.0f, 1 - uv.x));

        return CuDiff::wrap(x, y, z);
    }

    template<typename U>
    ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto pdf(const U& result)
    {
        auto [x, y, z] = CuDiff::unwrap(result);
        return z / glm::pi<float>();
    }
};

template<typename T>
struct SamplingStrategy<SamplingStrategyType::SPHERE_UNIFORM, T>
{
    ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto sample(const glm::vec2& uv)
    {
        auto z   = 1.0f - 2.0f * uv.x;
        auto r   = CuDiff::sqrt(CuDiff::max(0.0f, 1.0f - z * z));
        auto phi = 2.0f * glm::pi<float>() * uv.y;

        auto x = r * CuDiff::cos(phi);
        auto y = r * CuDiff::sin(phi);

        return CuDiff::wrap(x, y, z);
    }

    ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto pdf(const glm::vec3& /*result*/)
    {
        return 1.0f / (4.0f * glm::pi<float>());
    }
};
}    // namespace atcg