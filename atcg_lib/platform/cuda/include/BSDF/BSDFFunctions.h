#pragma once

#include <Core/glm.h>

#include <DataStructure/SampledSpectrum.h>

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
ATCG_HOST_DEVICE ATCG_FORCE_INLINE float D_GGX(const float NdotH, const float roughness)
{
    float a2 = roughness * roughness;
    float d  = (NdotH * a2 - NdotH) * NdotH + 1.0f;
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
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_HOST_DEVICE float fresnel_schlick(const float F0, const float VdotH)
{
    return F0 + (1.0f - F0) * glm::pow(glm::max(0.0f, 1.0f - VdotH), 5.0f);
}

/**
 * @brief Fresnel schlick approximation
 *
 * @param F0 The base reflectance at normal incidence
 * @param VdotH Angle between viewing direction and halfway vector
 *
 * @return Reflectance
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec3 fresnel_schlick(const glm::vec3& F0, const float VdotH)
{
    return F0 + (glm::vec3(1.0f) - F0) * glm::pow(glm::max(0.0f, 1.0f - VdotH), 5.0f);
}

/**
 * @brief Fresnel schlick approximation
 *
 * @param F0 The base reflectance at normal incidence
 * @param VdotH Angle between viewing direction and halfway vector
 *
 * @return Reflectance
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE atcg::SampledSpectrum fresnel_schlick(const atcg::SampledSpectrum& F0,
                                                                         const float VdotH)
{
    return F0 + (atcg::SampledSpectrum(1.0f) - F0) * glm::pow(glm::max(0.0f, 1.0f - VdotH), 5.0f);
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T V_SmithGGX(T NdotL, T NdotV, T alpha, T eps = 1e-8f)
{
    T a2      = alpha * alpha;
    T lambdaV = NdotL * glm::sqrt(NdotV * NdotV * (T(1) - a2) + a2);
    T lambdaL = NdotV * glm::sqrt(NdotL * NdotL * (T(1) - a2) + a2);
    return T(0.5) / (lambdaV + lambdaL + eps);
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE float G_SmithJointGGX(float NdotL, float NdotV, float roughness)
{
    float a2      = roughness * roughness;
    float LambdaL = 0.5f * (-1 + glm::sqrt(1.0f + a2 * (1 - NdotL * NdotL) / (NdotL * NdotL)));
    float LambdaV = 0.5f * (-1 + glm::sqrt(1.0f + a2 * (1 - NdotV * NdotV) / (NdotV * NdotV)));
    return 1.0f / (1.0f + LambdaL + LambdaV);
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec3 faceForward(const glm::vec3& normal, const glm::vec3& direction)
{
    return glm::dot(normal, direction) < 0.0f ? -normal : normal;
}
}