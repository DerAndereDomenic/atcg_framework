#pragma once

#include <Core/glm.h>
#include <Core/CUDA.h>

namespace atcg
{
template<int N>
struct PowerHeuristic
{
    template<typename T>
    static ATCG_HOST_DEVICE ATCG_INLINE float apply(const T& pdf_a, const T& pdf_b)
    {
        float power_a = glm::pow(pdf_a, N);
        float power_b = glm::pow(pdf_b, N);
        return power_a / (power_a + power_b);
    }
};

template<>
struct PowerHeuristic<1>
{
    template<typename T>
    static ATCG_HOST_DEVICE ATCG_INLINE float apply(const T& pdf_a, const T& pdf_b)
    {
        float power_a = pdf_a;
        float power_b = pdf_b;
        return power_a / (power_a + power_b);
    }
};

template<>
struct PowerHeuristic<2>
{
    template<typename T>
    static ATCG_HOST_DEVICE ATCG_INLINE float apply(const T& pdf_a, const T& pdf_b)
    {
        float power_a = pdf_a * pdf_a;
        float power_b = pdf_b * pdf_b;
        return power_a / (power_a + power_b);
    }
};

using BalanceHeuristic = PowerHeuristic<1>;
}    // namespace atcg