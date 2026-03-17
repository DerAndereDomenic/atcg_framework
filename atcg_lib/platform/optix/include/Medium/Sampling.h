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
}    // namespace atcg