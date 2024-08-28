#pragma once

#include <Math/Random.h>
#include <Math/Functions.h>
#include <Pathtracing/BSDFFLags.h>
#include <Pathtracing/SurfaceInteraction.h>
#include <Pathtracing/DirectCall.h>

namespace atcg
{

struct BSDFSamplingResult
{
    glm::vec3 out_dir;
    glm::vec3 bsdf_weight;
    float sample_probability = 0.0f;
};

struct BSDFEvalResult
{
    glm::vec3 bsdf_value     = glm::vec3(0);
    float sample_probability = 0.0f;
};

struct PBRBSDFData
{
    atcg::textureObject diffuse_texture;
    atcg::textureObject metallic_texture;
    atcg::textureObject roughness_texture;
};

struct RefractiveBSDFData
{
    atcg::textureObject diffuse_texture;
    float ior;
};

struct BSDFVPtrTable
{
    uint32_t sampleCallIndex;
    uint32_t evalCallIndex;

    BSDFComponentType flags;

#ifdef ATCG_RT_MODULE
    BSDFSamplingResult sampleBSDF(const SurfaceInteraction& si, PCG32& rng) const
    {
        return directCall<BSDFSamplingResult, const SurfaceInteraction&, PCG32&>(sampleCallIndex, si, rng);
    }

    BSDFEvalResult evalBSDF(const SurfaceInteraction& si, const glm::vec3& outgoing_dir) const
    {
        return directCall<BSDFEvalResult, const SurfaceInteraction&, const glm::vec3&>(evalCallIndex, si, outgoing_dir);
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

/**
 * @brief Schlick geometry term for GGX model
 *
 * @param NdotV Angle between normal and viewing direction
 * @param roughness The surface roughness
 *
 * @return The geometry term
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE float geometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0f);
    float k = (r * r) / 8.0f;

    float nom   = NdotV;
    float denom = NdotV * (1.0f - k) + k + 1e-5f;

    return nom / denom;
}

/**
 * @brief Smith geometry term
 *
 * @param NdotV Angle between normal and viewing direction
 * @param roughness The surface roughness
 *
 * @return The geometry term
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE float geometrySmith(float NdotL, float NdotV, float roughness)
{
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

/**
 * @brief Sample a pbr bdf
 *
 * @param si The surface interaction
 * @param diffuse_color The diffuse color
 * @param specular_F0 The specular reflectance (metallic color)
 * @param metallic The metallic parameter
 * @param roughness The surface roughness
 * @param rng The rng
 *
 * @return The sampling result
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE BSDFSamplingResult samplePBR(const SurfaceInteraction& si,
                                                                const glm::vec3& diffuse_color,
                                                                const glm::vec3& specular_F0,
                                                                const float& metallic,
                                                                const float& roughness,
                                                                PCG32& rng)
{
    BSDFSamplingResult result;

    // Direction towards viewer
    glm::vec3 view_dir = -si.incoming_direction;
    glm::vec3 normal   = si.normal;

    // Don't trace a new ray if surface is viewed from below
    float NdotV = glm::dot(normal, view_dir);
    if(NdotV <= 0)
    {
        return result;
    }

    // The matrix local_frame transforms a vector from the coordinate system where geom.N corresponds to the z-axis to
    // the world coordinate system.
    glm::mat3 local_frame = Math::compute_local_frame(normal);

    float diffuse_probability = glm::dot(diffuse_color, glm::vec3(1)) /
                                (glm::dot(diffuse_color, glm::vec3(1)) + glm::dot(specular_F0, glm::vec3(1)) + 1e-5f);
    float specular_probability = 1 - diffuse_probability;

    if(rng.next1d() < diffuse_probability)
    {
        // Sample light direction from diffuse bsdf
        glm::vec3 local_outgoing_ray_dir = warp_square_to_hemisphere_cosine(rng.next2d());
        // Transform local outgoing direction from tangent space to world space
        result.out_dir = local_frame * local_outgoing_ray_dir;
    }
    else
    {
        // Sample light direction from specular bsdf
        glm::vec3 local_halfway = warp_square_to_hemisphere_ggx(rng.next2d(), roughness);
        // Transform local halfway vector from tangent space to world space
        glm::vec3 halfway = local_frame * local_halfway;
        result.out_dir    = glm::reflect(si.incoming_direction, halfway);
    }

    // It is possible that light directions below the horizon are sampled..
    // If outgoing ray direction is below horizon, let the sampling fail!
    float NdotL = glm::dot(normal, result.out_dir);
    if(NdotL <= 0)
    {
        result.sample_probability = 0;
        return result;
    }

    glm::vec3 diffuse_bsdf = diffuse_color / glm::pi<float>();
    float diffuse_pdf      = NdotL / glm::pi<float>();

    glm::vec3 specular_bsdf = glm::vec3(0);
    float specular_pdf      = 0;
    // Only compute specular component if specular_f0 is not zero!
    glm::vec3 kD(1.0f);
    if(glm::dot(specular_F0, specular_F0) > 1e-6f)
    {
        glm::vec3 halfway = glm::normalize(result.out_dir + view_dir);
        float HdotV       = glm::dot(halfway, result.out_dir);
        float NdotH       = glm::dot(halfway, normal);

        // Normal distribution
        float NDF = D_GGX(NdotH, roughness);

        // Visibility
        float G = geometrySmith(NdotL, NdotV, roughness);

        // Fresnel
        glm::vec3 F = fresnel_schlick(specular_F0, HdotV);

        kD = (1.0f - F) * (1.0f - metallic);

        glm::vec3 numerator = NDF * G * F;
        float denominator   = 4.0f * NdotV * NdotL + 1e-5f;
        specular_bsdf       = numerator / denominator;

        float halfway_pdf = NDF * NdotH;
        float halfway_to_outgoing_pdf =
            warp_normal_to_reflected_direction_pdf(result.out_dir, halfway);    // 1 / (4*HdotV)
        specular_pdf = halfway_pdf * halfway_to_outgoing_pdf;
    }

    result.sample_probability = diffuse_probability * diffuse_pdf + specular_probability * specular_pdf + 1e-5f;
    result.bsdf_weight        = (specular_bsdf + kD * diffuse_bsdf) * NdotL / result.sample_probability;

    return result;
}

/**
 * @brief Evaluate a pbr bsdf
 *
 * @param si The surface interaction
 * @param outgoing_dir The outgoing direction
 * @param diffuse_color The diffuse color
 * @param metallic_color The metallic color
 * @param roughness The surface roughness
 * @param metallic The metallic value
 *
 * @return The eval result
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE BSDFEvalResult evalPBR(const SurfaceInteraction& si,
                                                          const glm::vec3& outgoing_dir,
                                                          const glm::vec3& diffuse_color,
                                                          const glm::vec3& metallic_color,
                                                          const float roughness,
                                                          const float metallic)
{
    atcg::BSDFEvalResult result;

    glm::vec3 light_dir = outgoing_dir;
    glm::vec3 view_dir  = -si.incoming_direction;

    glm::vec3 H = glm::normalize(light_dir + view_dir);

    float NdotH = glm::max(glm::dot(si.normal, H), 0.0f);
    float NdotV = glm::max(glm::dot(si.normal, view_dir), 0.0f);
    float NdotL = glm::max(glm::dot(si.normal, light_dir), 0.0f);

    if(NdotL <= 0.0f || NdotV <= 0.0f) return result;

    float NDF   = atcg::D_GGX(NdotH, roughness);
    float G     = atcg::geometrySmith(NdotL, NdotV, roughness);
    glm::vec3 F = atcg::fresnel_schlick(metallic_color, glm::max(glm::dot(H, view_dir), 0.0f));

    glm::vec3 numerator = NDF * G * F;
    float denominator   = 4.0 * NdotV * NdotL + 1e-5f;
    glm::vec3 specular  = numerator / denominator;

    glm::vec3 kS = F;
    glm::vec3 kD = glm::vec3(1.0) - kS;
    kD *= (1.0 - metallic);

    float diffuse_probability =
        glm::dot(diffuse_color, glm::vec3(1)) /
        (glm::dot(diffuse_color, glm::vec3(1)) + glm::dot(metallic_color, glm::vec3(1)) + 1e-5f);
    float specular_probability    = 1 - diffuse_probability;
    float diffuse_pdf             = NdotL / glm::pi<float>();
    float halfway_pdf             = NDF * NdotH;
    float halfway_to_outgoing_pdf = atcg::warp_normal_to_reflected_direction_pdf(outgoing_dir, H);    // 1 / (4*HdotV)
    float specular_pdf            = halfway_pdf * halfway_to_outgoing_pdf;

    result.bsdf_value         = specular + kD * diffuse_color / glm::pi<float>();
    result.sample_probability = diffuse_probability * diffuse_pdf + specular_probability * specular_pdf + 1e-5f;

    return result;
}

/**
 * @brief Sample a refractive BSDF
 *
 * @param si The surface interaction
 * @param diffuse_color The diffuse color
 * @param ior The index of refraction
 * @param rng The rng
 *
 * @return The sampling result
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE BSDFSamplingResult sampleRefractive(const SurfaceInteraction& si,
                                                                       const glm::vec3& diffuse_color,
                                                                       const float ior,
                                                                       PCG32& rng)
{
    // Determine surface parameters
    bool outsidein             = glm::dot(si.incoming_direction, si.normal) < 0;
    glm::vec3 interface_normal = outsidein ? si.normal : -si.normal;
    float eta                  = outsidein ? 1.0f / ior : ior;

    // Compute outgoing ray directions
    glm::vec3 transmitted_ray_dir = glm::refract(si.incoming_direction, interface_normal, eta);
    glm::vec3 reflected_ray_dir   = glm::reflect(si.incoming_direction, interface_normal);

    // Fresnel reflectance at normal incidence
    float F0 = (eta - 1) / (eta + 1);
    F0       = F0 * F0;

    float NdotL = glm::abs(glm::dot(si.incoming_direction, interface_normal));

    // Reflection an transmission probabilities
    float reflection_probability   = atcg::fresnel_schlick(F0, NdotL);
    float transmission_probability = 1.0f - reflection_probability;
    if(glm::dot(transmitted_ray_dir, transmitted_ray_dir) < 1e-6f)
    {
        // Total internal reflection!
        transmission_probability = 0.0f;
        reflection_probability   = 1.0f;
    }


    // Compute sampling result
    atcg::BSDFSamplingResult result;
    result.sample_probability = 0;

    // Stochastically select a reflection or transmission via russian roulette
    if(rng.next1d() < reflection_probability)
    {
        // Select the reflection event
        // We sample the BDSF exactly.
        result.bsdf_weight        = diffuse_color;
        result.out_dir            = reflected_ray_dir;
        result.sample_probability = reflection_probability;
    }
    else
    {
        // Select the transmission event
        // We sample the BDSF exactly.
        result.bsdf_weight        = diffuse_color;
        result.out_dir            = transmitted_ray_dir;
        result.sample_probability = transmission_probability;
    }

    return result;
}

/**
 * @brief Evaluate a refractive bsdf
 *
 * @param si The surface interaction
 * @param outgoing_dir The outgoing direction
 *
 * @return The eval result
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE BSDFEvalResult evalRefractive(const SurfaceInteraction& si,
                                                                 const glm::vec3& outgoing_dir)
{
    // TODO: Return empty result for now because most of the time a randomly sampled direction will not hit the delta
    // function of a perfect reflection/transmission
    return atcg::BSDFEvalResult();
}
}    // namespace atcg