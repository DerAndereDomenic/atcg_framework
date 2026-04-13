#pragma once

#include <BSDF/Sampling.h>

namespace atcg
{
template<typename T>
struct SamplingStrategy<SamplingStrategyType::EXPONENTIAL_SAMPLING, T>
{
    T _density;
    ATCG_HOST_DEVICE ATCG_FORCE_INLINE SamplingStrategy<SamplingStrategyType::EXPONENTIAL_SAMPLING, T>(T density)
        : _density(density)
    {
    }

    ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto sample(const T u) { return -CuDiff::log(u) / _density; }

    ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto pdf(const T t) { return _density * CuDiff::exp(-_density * t); }
};

template<typename T>
struct SamplingStrategy<SamplingStrategyType::HG_PHASE, T>
{
    T _g;
    ATCG_HOST_DEVICE ATCG_FORCE_INLINE SamplingStrategy<SamplingStrategyType::HG_PHASE, T>(T g) : _g(g) {}

    ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto sample(const glm::vec2& uv)
    {
        if(_g == 0.0f)
        {
            atcg::SamplingStrategy<SamplingStrategyType::SPHERE_UNIFORM, T> uniform_strategy;
            return uniform_strategy.sample(uv);
        }

        float u1 = uv.x;
        float u2 = uv.y;

        float g2        = _g * _g;
        float d         = (1 - g2) / (1 - _g + 2 * _g * u1);
        float cos_theta = 0.5 / _g * (1 + g2 - d * d);

        float sin_theta = glm::sqrt(glm::max(0.0f, 1.0f - cos_theta * cos_theta));
        float phi       = 2 * glm::pi<float>() * u2;

        float x = sin_theta * glm::cos(phi);
        float y = sin_theta * glm::sin(phi);
        float z = cos_theta;

        return glm::vec3(x, y, z);
    }

    template<typename U>
    ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto pdf(const U cos_theta)
    {
        T g2       = _g * _g;
        float area = 4 * glm::pi<float>();    // area of sphere
        auto phase = (1 - g2) / area * CuDiff::pow((1 + g2 - 2 * _g * cos_theta), -1.5f);
        return phase;
    }
};

template<typename T>
struct SamplingStrategy<SamplingStrategyType::RAYLEIGH_PHASE, T>
{
    ATCG_HOST_DEVICE ATCG_FORCE_INLINE SamplingStrategy<SamplingStrategyType::RAYLEIGH_PHASE, T>() {}

    ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto sample(const glm::vec2& uv)
    {
        float xi1 = uv.x;
        float xi2 = uv.y;

        float a = 4.0f * xi1 - 2.0f;
        float b = glm::sqrt(a * a + 1.0f);

        float mu = cbrt(a + b) + cbrt(a - b);

        float phi = glm::two_pi<float>() * xi2;

        float sin_theta = glm::sqrt(glm::max(0.0f, 1.0f - mu * mu));

        return glm::vec3(sin_theta * glm::cos(phi), sin_theta * glm::sin(phi), mu);
    }

    ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto pdf(const float cos_theta)
    {
        float phase = 3.0f / (16.0f * glm::pi<float>()) * (1 + cos_theta * cos_theta);
        return phase;
    }
};
}    // namespace atcg