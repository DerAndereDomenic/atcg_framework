#pragma once

#include <BSDF/Sampling.h>

namespace atcg
{
template<>
struct SamplingStrategy<SamplingStrategyType::EXPONENTIAL_SAMPLING>
{
    float _density;
    ATCG_HOST_DEVICE ATCG_FORCE_INLINE SamplingStrategy<SamplingStrategyType::EXPONENTIAL_SAMPLING>(float density)
        : _density(density)
    {
    }

    ATCG_HOST_DEVICE ATCG_FORCE_INLINE float sample(const float u) { return -glm::log(u) / _density; }

    ATCG_HOST_DEVICE ATCG_FORCE_INLINE float pdf(const float t) { return _density * glm::exp(-_density * t); }
};
}    // namespace atcg