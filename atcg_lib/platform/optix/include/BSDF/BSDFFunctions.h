#pragma once

#include <Core/glm.h>

#include <Spectrum/SampledSpectrum.h>

namespace atcg
{
/**
 * @brief Normal distribution function of GGX microfacet model
 *
 * @param NdotH The angle between normal and half way vector
 * @param roughness
 *
 * @return The pdf value
 */
template<typename NdotHType, typename roughnessType>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto D_GGX(const NdotHType& NdotH, const roughnessType& roughness)
{
    auto a2 = roughness * roughness;
    auto d  = (NdotH * a2 - NdotH) * NdotH + 1.0f;
    return a2 / (glm::pi<float>() * d * d + 1e-5f);
}

/**
 * @brief Fresnel schlick approximation
 *
 * @param F0 The base reflectance at normal incidence
 * @param VdotH Angle between viewing direction and halfway vector
 *
 * @return Reflectance
 */
template<typename F0Type, typename VdotHType>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto fresnel_schlick(const F0Type& F0, const VdotHType& VdotH)
{
    return F0 +
           (F0Type(CuDiff::dual_value_type_t<F0Type>(1.0f)) - F0) * CuDiff::pow(CuDiff::max(0.0f, 1.0f - VdotH), 5.0f);
}

template<typename NdotLType, typename NdotVType, typename alphaType>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto
V_SmithGGX(const NdotLType& NdotL, const NdotVType& NdotV, const alphaType& alpha, float eps = 1e-8f)
{
    auto a2      = alpha * alpha;
    auto lambdaV = NdotL * CuDiff::sqrt(NdotV * NdotV * (1.0f - a2) + a2);
    auto lambdaL = NdotV * CuDiff::sqrt(NdotL * NdotL * (1.0f - a2) + a2);
    return 0.5f / (lambdaV + lambdaL + eps);
}

template<typename NdotLType, typename NdotVType, typename roughnessType>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto G_SmithJointGGX(NdotLType NdotL, NdotVType NdotV, roughnessType roughness)
{
    auto a2      = roughness * roughness;
    auto LambdaL = 0.5f * (-1.0f + CuDiff::sqrt(1.0f + a2 * (1.0f - NdotL * NdotL) / (NdotL * NdotL)));
    auto LambdaV = 0.5f * (-1.0f + CuDiff::sqrt(1.0f + a2 * (1.0f - NdotV * NdotV) / (NdotV * NdotV)));
    return 1.0f / (1.0f + LambdaL + LambdaV);
}
}