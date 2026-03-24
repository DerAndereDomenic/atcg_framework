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

    ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto sample(const T u) { return -glm::log(u) / _density; }

    ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto pdf(const T t) { return _density * glm::exp(-_density * t); }
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

    ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto pdf(const glm::vec3& result_local)
    {
        float cos_theta = result_local.z;
        float g2        = _g * _g;
        float area      = 4 * glm::pi<float>();    // area of sphere
        float phase     = (1 - g2) / area * glm::pow((1 + g2 - 2 * _g * cos_theta), -1.5f);
        return phase;
    }
};
}    // namespace atcg