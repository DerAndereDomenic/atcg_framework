#pragma once

#include <Core/glm.h>
#include <Core/CUDA.h>
#include <Pathtracing/BSDFFlags.h>
#include <Pathtracing/SurfaceInteraction.h>
#include <Math/Random.h>

namespace atcg
{
class BSDF
{
public:
    BSDF() = default;

    virtual ~BSDF() {}

protected:
    BSDFComponentType _flags;
};

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

inline ATCG_HOST_DEVICE glm::mat3 compute_local_frame(const glm::vec3& localZ)
{
    float x  = localZ.x;
    float y  = localZ.y;
    float z  = localZ.z;
    float sz = (z >= 0) ? 1 : -1;
    float a  = 1 / (sz + z);
    float ya = y * a;
    float b  = x * ya;
    float c  = x * sz;

    glm::vec3 localX = glm::vec3(c * x * a - 1, sz * b, c);
    glm::vec3 localY = glm::vec3(b, y * ya - sz, y);

    glm::mat3 frame;
    // Set columns of matrix
    frame[0] = localX;
    frame[1] = localY;
    frame[2] = localZ;
    return frame;
}

inline ATCG_HOST_DEVICE float D_GGX(const float NdotH, const float roughness)
{
    float a2 = roughness * roughness;
    float d  = (NdotH * a2 - NdotH) * NdotH + 1.0f;
    return a2 / (glm::pi<float>() * d * d + 1e-5f);
}

inline ATCG_HOST_DEVICE glm::vec3 warp_square_to_hemisphere_ggx(const glm::vec2& uv, float roughness)
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

inline ATCG_HOST_DEVICE float warp_square_to_hemisphere_ggx_pdf(const glm::vec3& result, float roughness)
{
    return D_GGX(result.z, roughness) * glm::max(0.0f, result.z);
}


inline ATCG_HOST_DEVICE glm::vec3 warp_square_to_hemisphere_cosine(const glm::vec2& uv)
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

inline ATCG_HOST_DEVICE float warp_square_to_hemisphere_cosine_pdf(const glm::vec3& result)
{
    return glm::max(0.0f, result.z) / glm::pi<float>();
}

inline ATCG_HOST_DEVICE float warp_normal_to_reflected_direction_pdf(const glm::vec3& reflected_dir,
                                                                     const glm::vec3& normal)
{
    return 1 / glm::abs(4 * glm::dot(reflected_dir, normal));
}

inline ATCG_HOST_DEVICE float fresnel_schlick(const float F0, const float VdotH)
{
    return F0 + (1.0f - F0) * glm::pow(glm::max(0.0f, 1.0f - VdotH), 5.0f);
}

inline ATCG_HOST_DEVICE glm::vec3 fresnel_schlick(const glm::vec3& F0, const float VdotH)
{
    return F0 + (glm::vec3(1.0f) - F0) * glm::pow(glm::max(0.0f, 1.0f - VdotH), 5.0f);
}

inline ATCG_HOST_DEVICE float geometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0f);
    float k = (r * r) / 8.0f;

    float nom   = NdotV;
    float denom = NdotV * (1.0f - k) + k + 1e-5f;

    return nom / denom;
}

inline ATCG_HOST_DEVICE float geometrySmith(float NdotL, float NdotV, float roughness)
{
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

inline ATCG_HOST_DEVICE float rgb2scalar(const glm::vec3& rgb)
{
    return (rgb.x + rgb.y + rgb.z) / 3.0f;
}

inline ATCG_HOST_DEVICE BSDFSamplingResult sampleGGX(const SurfaceInteraction& si,
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
    glm::mat3 local_frame = compute_local_frame(normal);

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

}    // namespace atcg