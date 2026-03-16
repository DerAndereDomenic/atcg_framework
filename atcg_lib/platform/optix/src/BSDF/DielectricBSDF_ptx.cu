#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>

#include <Math/Functions.h>
#include <Core/SurfaceInteraction.h>
#include <BSDF/BSDFVPtrTable.cuh>
#include <BSDF/DielectricBSDFData.cuh>
#include <BSDF/BSDFFunctions.h>
#include <BSDF/Sampling.h>

#include <DataStructure/Frame.h>

namespace detail
{

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
ATCG_HOST_DEVICE ATCG_FORCE_INLINE atcg::BSDFSamplingResult
sampleRefractive(const atcg::SurfaceInteraction& si,
                 const atcg::SampledSpectrum& reflectance_color,
                 const float roughness,
                 const float ior,
                 atcg::PCG32& rng)
{
    glm::vec3 wi = -si.incoming_direction;

    // Determine surface parameters
    bool outsidein             = glm::dot(wi, si.normal) > 0;
    glm::vec3 interface_normal = outsidein ? si.normal : -si.normal;
    float eta                  = outsidein ? 1.0f / ior : ior;

    atcg::Frame local_frame = atcg::Frame(interface_normal);

    atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_GGX> strategy(roughness);
    glm::vec3 local_halfway = strategy.sample(rng.next2d());
    float halfway_pdf       = strategy.pdf(local_halfway);
    // Transform local halfway vector from tangent space to world space
    glm::vec3 halfway = local_frame.toWorld(local_halfway);

    // Compute outgoing ray directions
    glm::vec3 transmitted_ray_dir = glm::refract(-wi, halfway, eta);
    glm::vec3 reflected_ray_dir   = glm::reflect(-wi, halfway);

    // Fresnel reflectance at normal incidence
    float F0 = (eta - 1.0f) / (eta + 1.0f);
    F0       = F0 * F0;

    // Reflection an transmission probabilities
    float HdotV                    = glm::dot(wi, halfway);
    float F                        = atcg::fresnel_schlick(F0, HdotV);
    float reflection_probability   = F;
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
    glm::vec3 wo;
    float NdotL;
    float HdotL;
    if(rng.next1d() < reflection_probability)
    {
        wo = reflected_ray_dir;
        float light_dir_pdf =
            halfway_pdf *
            atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_GGX>::warp_halfway_to_reflected_direction_pdf(
                wo,
                halfway) *
            reflection_probability;

        result.sample_probability = light_dir_pdf;
        NdotL                     = glm::dot(interface_normal, wo);
        HdotL                     = glm::dot(halfway, wo);

        result.flags =
            roughness < 0.1f ? atcg::BSDFComponentType::IdealReflection : atcg::BSDFComponentType::GlossyReflection;
    }
    else
    {
        wo    = transmitted_ray_dir;
        HdotL = glm::dot(halfway, wo);
        float light_dir_pdf =
            halfway_pdf *
            atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_GGX>::warp_halfway_to_refracted_direction_pdf(
                HdotV,
                HdotL,
                eta) *
            transmission_probability;

        result.sample_probability = light_dir_pdf;
        NdotL                     = -glm::dot(interface_normal, wo);

        result.flags =
            roughness < 0.1f ? atcg::BSDFComponentType::IdealTransmission : atcg::BSDFComponentType::GlossyTransmission;
    }

    if(NdotL <= 0)
    {
        result.sample_probability = 0;
        return result;
    }

    float NdotV = glm::abs(glm::dot(interface_normal, wi));
    float NdotH = glm::dot(halfway, interface_normal);

    float G            = atcg::G_SmithJointGGX(NdotL, NdotV, roughness);
    result.bsdf_weight = reflectance_color * G * glm::abs(HdotL) / (NdotV * NdotH);
    result.out_dir     = wo;

    return result;
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE atcg::BSDFEvalResult evalRefractive(const atcg::SurfaceInteraction& si,
                                                                       const glm::vec3& outgoing_dir,
                                                                       const atcg::SampledSpectrum& reflectance_color,
                                                                       const float roughness,
                                                                       const float ior)
{
    atcg::BSDFEvalResult result;

    glm::vec3 wo = outgoing_dir;
    glm::vec3 wi = -si.incoming_direction;

    bool outsidein = glm::dot(wi, si.normal) > 0;
    float eta      = outsidein ? 1.0f / ior : ior;

    bool outsideout = glm::dot(wo, si.normal) > 0;

    bool same_side = outsidein == outsideout;

    atcg::SampledSpectrum specular_bsdf = atcg::SampledSpectrum(0);

    float F0 = (eta - 1) / (eta + 1);
    F0       = F0 * F0;

    glm::vec3 interface_normal = outsidein ? si.normal : -si.normal;
    float light_dir_pdf        = 0.0f;
    if(same_side)
    {
        glm::vec3 halfway = glm::normalize(wi + wo);
        float NdotH       = glm::dot(halfway, interface_normal);
        float LdotH       = glm::dot(halfway, wo);

        float NdotL = glm::dot(interface_normal, wo);
        float NdotV = glm::dot(interface_normal, wi);

        float D = atcg::D_GGX(NdotH, roughness);
        float G = atcg::G_SmithJointGGX(NdotL, NdotV, roughness);

        glm::vec3 refracted = glm::refract(-wi, halfway, eta);
        float F             = 1.0f;
        if(glm::length2(refracted) > 1e-6f)
        {
            F = atcg::fresnel_schlick(F0, LdotH);
        }
        float reflection_probability = F;

        light_dir_pdf =
            D * NdotH * reflection_probability *
            atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_GGX>::warp_halfway_to_reflected_direction_pdf(
                wo,
                halfway);

        specular_bsdf = reflectance_color * D * G * F / (4.0f * NdotV * NdotL + 1e-5f);
        result.flags =
            roughness < 0.1f ? atcg::BSDFComponentType::IdealReflection : atcg::BSDFComponentType::GlossyReflection;
    }
    else
    {
        glm::vec3 halfway = -glm::normalize(eta * wi + wo);
        // The halfway vector always points into the thinner medium
        glm::vec3 thin_normal = ior > 1.0f ? si.normal : -si.normal;
        float NdotH           = glm::dot(si.normal, halfway);

        float LdotH = glm::dot(wo, halfway);
        float VdotH = glm::dot(wi, halfway);

        float NdotL = glm::abs(glm::dot(si.normal, wo));
        float NdotV = glm::abs(glm::dot(si.normal, wi));

        float D = atcg::D_GGX(NdotH, roughness);
        float G = atcg::G_SmithJointGGX(NdotL, NdotV, roughness);

        float F                = atcg::fresnel_schlick(F0, glm::abs(VdotH));
        float T                = 1.0f - F;
        float transmission_pdf = T;

        float denom = (LdotH + eta * VdotH);
        denom *= denom;

        float numerator = eta * eta * T * D * G * glm::abs(LdotH) * glm::abs(VdotH);

        light_dir_pdf =
            D * NdotH * transmission_pdf *
            atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_GGX>::warp_halfway_to_refracted_direction_pdf(
                VdotH,
                LdotH,
                eta);

        specular_bsdf = reflectance_color * numerator / (denom * NdotL * NdotV + 1e-5f);

        result.flags =
            roughness < 0.1f ? atcg::BSDFComponentType::IdealTransmission : atcg::BSDFComponentType::GlossyTransmission;
    }

    result.bsdf_value         = specular_bsdf * glm::abs(glm::dot(si.normal, wo));
    result.sample_probability = light_dir_pdf;
    return result;
}

}    // namespace detail

extern "C" __device__ atcg::BSDFSamplingResult
__direct_callable__sample_dielectricbsdf(const atcg::SurfaceInteraction& si,
                                         const atcg::SampledWavelengths& wavelengths,
                                         atcg::PCG32& rng)
{
    const atcg::DielectricBSDFData* sbt_data =
        *reinterpret_cast<const atcg::DielectricBSDFData**>(optixGetSbtDataPointer());

    atcg::SampledSpectrum reflectance_color =
        atcg::SampledSpectrum::fromRGB(sbt_data->diffuse_texture.read(si.uv), wavelengths);
    float roughness = sbt_data->roughness_texture.read(si.uv);
    roughness       = glm::max(roughness * roughness, 1e-3f);    // In the real time shaders, roughness is squared

    float ior = sbt_data->ior_texture.read(si.uv);

    return detail::sampleRefractive(si, reflectance_color, roughness, ior, rng);
}

extern "C" __device__ atcg::BSDFEvalResult
__direct_callable__eval_dielectricbsdf(const atcg::SurfaceInteraction& si,
                                       const glm::vec3& outgoing_dir,
                                       const atcg::SampledWavelengths& wavelengths)
{
    const atcg::DielectricBSDFData* sbt_data =
        *reinterpret_cast<const atcg::DielectricBSDFData**>(optixGetSbtDataPointer());

    atcg::SampledSpectrum reflectance_color =
        atcg::SampledSpectrum::fromRGB(sbt_data->diffuse_texture.read(si.uv), wavelengths);
    float roughness = sbt_data->roughness_texture.read(si.uv);
    roughness       = glm::max(roughness * roughness, 1e-3f);    // In the real time shaders, roughness is squared

    float ior = sbt_data->ior_texture.read(si.uv);

    return detail::evalRefractive(si, outgoing_dir, reflectance_color, roughness, ior);
}