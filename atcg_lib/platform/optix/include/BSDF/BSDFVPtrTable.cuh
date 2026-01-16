#pragma once

#include <Core/glm.h>
#include <Core/SurfaceInteraction.h>
#include <Math/Random.h>
#include <BSDF/BSDFFlags.h>
#include <optix.h>

#include <CuDiff/Dual.h>

namespace atcg
{
struct BSDFSamplingResult
{
    glm::vec3 out_dir;
    glm::vec3 bsdf_weight;
    float sample_probability = 0.0f;
    BSDFComponentType flags  = BSDFComponentType::Any;
};

struct BSDFDualSamplingResult
{
    CuDiff::Dual<6, glm::vec3> out_dir;
    CuDiff::Dual<6, glm::vec3> bsdf_weight;
    CuDiff::Dual<6, float> sample_probability;
    BSDFComponentType flags = BSDFComponentType::Any;
};

struct BSDFEvalResult
{
    glm::vec3 bsdf_value     = glm::vec3(0);
    float sample_probability = 0.0f;
    BSDFComponentType flags  = BSDFComponentType::Any;
};

struct BSDFDualEvalResult
{
    CuDiff::Dual<4, glm::vec3> bsdf_value;
    CuDiff::Dual<4, float> sample_probability;
};

struct BSDFVPtrTable
{
    uint32_t sampleCallIndex;
    uint32_t evalCallIndex;
    uint32_t sampleForwardCallIndex;
    uint32_t evalForwardCallIndex;
    uint32_t sampleBackwardCallIndex;
    uint32_t evalBackwardCallIndex;


    BSDFComponentType flags;

#ifdef __CUDACC__

    __device__ BSDFSamplingResult sampleBSDF(const SurfaceInteraction& si, PCG32& rng) const
    {
        return optixDirectCall<BSDFSamplingResult, const SurfaceInteraction&, PCG32&>(sampleCallIndex, si, rng);
    }

    __device__ BSDFEvalResult evalBSDF(const SurfaceInteraction& si, const glm::vec3& outgoing_dir) const
    {
        return optixDirectCall<BSDFEvalResult, const SurfaceInteraction&, const glm::vec3&>(evalCallIndex,
                                                                                            si,
                                                                                            outgoing_dir);
    }

    __device__ BSDFDualSamplingResult sampleBSDFForward(const DualSurfaceInteraction& si, PCG32& rng) const
    {
        return optixDirectCall<BSDFDualSamplingResult, const DualSurfaceInteraction&, PCG32&>(sampleForwardCallIndex,
                                                                                              si,
                                                                                              rng);
    }

    __device__ BSDFDualEvalResult evalBSDFForward(const SurfaceInteraction& si, const glm::vec3& outgoing_dir) const
    {
        return optixDirectCall<BSDFDualEvalResult, const SurfaceInteraction&, const glm::vec3&>(evalForwardCallIndex,
                                                                                                si,
                                                                                                outgoing_dir);
    }

    __device__ void
    evalBSDFBackward(const SurfaceInteraction& si, const glm::vec3& outgoing_dir, const glm::vec3& out_grad) const
    {
        optixDirectCall<void, const SurfaceInteraction&, const glm::vec3&, const glm::vec3&>(evalBackwardCallIndex,
                                                                                             si,
                                                                                             outgoing_dir,
                                                                                             out_grad);
    }

    __device__ void
    sampleBSDFBackward(const SurfaceInteraction& si, PCG32& rng, const glm::vec3& dLdbsdf, const glm::vec2& dLdwo) const
    {
        optixDirectCall<void, const SurfaceInteraction&, PCG32&, const glm::vec3&, const glm::vec2&>(
            sampleBackwardCallIndex,
            si,
            rng,
            dLdbsdf,
            dLdwo);
    }

#endif
};

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
 * @brief Sample a direction according to the GGX normal distribution function
 *
 * @param uv The random numbers used for sampling
 * @param roughness The surface roughness
 *
 * @return The sampled direction
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec3 warp_square_to_hemisphere_ggx(const glm::vec2& uv, float roughness)
{
    // GGX NDF sampling
    float cos_theta = glm::sqrt((1.0f - uv.x) / (1.0f + (roughness * roughness - 1.0f) * uv.x));
    float sin_theta = glm::sqrt(glm::max(0.0f, 1.0f - cos_theta * cos_theta));
    float phi       = 2.0f * glm::pi<float>() * uv.y;

    float x = sin_theta * glm::cos(phi);
    float y = sin_theta * glm::sin(phi);
    float z = cos_theta;

    return glm::vec3(x, y, z);
}

/**
 * @brief Evaluate the pdf of sampling a direction according to the GGX normal distribution function
 *
 * @param result The direction
 * @param roughness The surface roughness
 *
 * @return The pdf result
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE float warp_square_to_hemisphere_ggx_pdf(const glm::vec3& result, float roughness)
{
    return D_GGX(result.z, roughness) * glm::max(0.0f, result.z);
}

/**
 * @brief Sample a direction in the hemisphere using cosine weighted sampling
 *
 * @param uv The random numbers used for sampling
 *
 * @return The direction
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec3 warp_square_to_hemisphere_cosine(const glm::vec2& uv)
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

/**
 * @brief Evaluate the pdf of sampling a direction according to a cosine weighted distribution
 *
 * @param result The direction
 *
 * @return The pdf result
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE float warp_square_to_hemisphere_cosine_pdf(const glm::vec3& result)
{
    return glm::max(0.0f, result.z) / glm::pi<float>();
}

/**
 * @brief Jacobian of transforming a halfway direction to reflected direction
 *
 * @param HdotV Dot product between halfway and incoming direction
 * @param HdotL Dot product between halfway and outgoing direction
 * @param eta Index of refraction n1/n2
 *
 * @return The pdf
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE float
warp_normal_to_refracted_direction_pdf(const float HdotV, const float HdotL, const float eta)
{
    float denom = (HdotL + eta * HdotV);
    return eta * eta * glm::abs(HdotV) / (denom * denom);
}

/**
 * @brief Jacobian of transforming a halfway direction to reflected direction
 *
 * @param reflected_dir The reflected direction
 * @param normal The surface normal
 *
 * @return The pdf
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE float warp_normal_to_reflected_direction_pdf(const glm::vec3& reflected_dir,
                                                                                const glm::vec3& normal)
{
    return 1 / glm::abs(4 * glm::dot(reflected_dir, normal));
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

}    // namespace atcg