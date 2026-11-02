#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>

#include <Utils/HostDevice.h>
#include <Core/SurfaceInteraction.h>
#include <BSDF/BSDFVPtrTable.cuh>
#include <BSDF/DielectricBSDFData.cuh>
#include <BSDF/BSDFFunctions.h>
#include <BSDF/Sampling.h>

#include <DataStructure/Frame.h>

namespace detail
{

ATCG_HOST_DEVICE ATCG_FORCE_INLINE float evalRefractivePDF(const atcg::SurfaceInteraction& si,
                                                           const glm::vec3& outgoing_dir,
                                                           const atcg::SampledSpectrum& reflectance_color,
                                                           const float roughness,
                                                           const float ior)
{
    glm::vec3 wi = -si.incoming_direction;    // Points towards the last traced vertex (might be the camera)
    glm::vec3 wo = outgoing_dir;              // Points towards next path vertex (might be a light source)

    // Convention: normals points into the thinner medium. If the ior is greater than 1, then the normal points into the
    // surface.
    glm::vec3 normal = ior > 1.0f ? si.normal : -si.normal;

    float cos_theta_i = glm::dot(normal, wi);
    float cos_theta_o = glm::dot(normal, wo);

    glm::vec3 interface_normal = cos_theta_i > 0.0f ? normal : -normal;

    bool reflected = cos_theta_i * cos_theta_o > 0.0f;

    float eta = cos_theta_i > 0.0f ? 1.0f / ior : ior;

    if(reflected)
    {
        glm::vec3 wh = glm::sign(cos_theta_i) * glm::normalize(wi + wo);

        float D = atcg::D_GGX(glm::abs(glm::dot(normal, wh)), roughness);
        float G = atcg::G_SmithJointGGX(glm::abs(cos_theta_o), glm::abs(cos_theta_i), roughness);

        float F = atcg::fresnel(eta, glm::abs(glm::dot(wi, wh)));

        float pdf_dw =
            D * glm::abs(glm::dot(normal, wh)) * F *
            atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_GGX>::warp_halfway_to_reflected_direction_pdf(
                wo,
                wh);

        return pdf_dw;
    }
    else
    {
        glm::vec3 wh = -glm::normalize(eta * wi + wo);

        if(glm::dot(wh, interface_normal) < 0.0f)
        {
            return 0.0f;
        }

        if(glm::dot(wi, wh) < 0.0f)
        {
            return 0.0f;
        }

        float D = atcg::D_GGX(glm::abs(glm::dot(normal, wh)), roughness);
        float G = atcg::G_SmithJointGGX(glm::abs(cos_theta_o), glm::abs(cos_theta_i), roughness);
        float F = atcg::fresnel(eta, glm::abs(glm::dot(wi, wh)));

        float denom = (eta * glm::dot(wi, wh) + glm::dot(wo, wh));
        float value = (1.0f - F) * G * D * glm::abs(glm::dot(wi, wh)) * glm::abs(glm::dot(wo, wh)) /
                      (denom * denom * glm::abs(cos_theta_i));
        float pdf_dw =
            D * glm::abs(glm::dot(normal, wh)) * (1.0f - F) *
            atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_GGX>::warp_halfway_to_refracted_direction_pdf(
                glm::dot(wi, wh),
                glm::dot(wo, wh),
                eta);

        return pdf_dw;
    }
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE atcg::BSDFEvalResult evalRefractive(const atcg::SurfaceInteraction& si,
                                                                       const glm::vec3& outgoing_dir,
                                                                       const atcg::SampledSpectrum& reflectance_color,
                                                                       const float roughness,
                                                                       const float ior,
                                                                       const atcg::BSDFContext& ctx)
{
    atcg::BSDFEvalResult result;

    glm::vec3 wi = -si.incoming_direction;    // Points towards the last traced vertex (might be the camera)
    glm::vec3 wo = outgoing_dir;              // Points towards next path vertex (might be a light source)

    // Convention: normals points into the thinner medium. If the ior is greater than 1, then the normal points into the
    // surface.
    glm::vec3 normal = ior > 1.0f ? si.normal : -si.normal;

    float cos_theta_i = glm::dot(normal, wi);
    float cos_theta_o = glm::dot(normal, wo);

    glm::vec3 interface_normal = cos_theta_i > 0.0f ? normal : -normal;

    bool reflected = cos_theta_i * cos_theta_o > 0.0f;

    float eta = cos_theta_i > 0.0f ? 1.0f / ior : ior;

    float value = 0.0f;
    if(reflected)
    {
        glm::vec3 wh = glm::sign(cos_theta_i) * glm::normalize(wi + wo);

        float D = atcg::D_GGX(glm::abs(glm::dot(normal, wh)), roughness);
        float G = atcg::G_SmithJointGGX(glm::abs(cos_theta_o), glm::abs(cos_theta_i), roughness);

        float F = atcg::fresnel(eta, glm::abs(glm::dot(wi, wh)));

        value = F * D * G / (4.0f * glm::abs(cos_theta_i));
        result.pdf_dw =
            D * glm::abs(glm::dot(normal, wh)) * F *
            atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_GGX>::warp_halfway_to_reflected_direction_pdf(
                wo,
                wh);
        result.flags =
            roughness < 0.01f ? atcg::BSDFComponentType::IdealReflection : atcg::BSDFComponentType::GlossyReflection;
    }
    else
    {
        glm::vec3 wh = -glm::sign(cos_theta_i) * glm::normalize(eta * wi + wo);

        if(glm::dot(wi, wh) < 0.0f)
        {
            return result;
        }

        float D = atcg::D_GGX(glm::abs(glm::dot(normal, wh)), roughness);
        float G = atcg::G_SmithJointGGX(glm::abs(cos_theta_o), glm::abs(cos_theta_i), roughness);
        float F = atcg::fresnel(eta, glm::abs(glm::dot(wi, wh)));

        float scale = ctx.transport_mode == atcg::TransportMode::IMPORTANCE ? 1.0f : (eta * eta);

        float denom = (eta * glm::dot(wi, wh) + glm::dot(wo, wh));
        value       = scale * (1.0f - F) * G * D * glm::abs(glm::dot(wi, wh)) * glm::abs(glm::dot(wo, wh)) /
                      (denom * denom * glm::abs(cos_theta_i));
        result.pdf_dw =
            D * glm::abs(glm::dot(normal, wh)) * (1.0f - F) *
            atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_GGX>::warp_halfway_to_refracted_direction_pdf(
                glm::dot(wi, wh),
                glm::dot(wo, wh),
                eta);

        result.flags = roughness < 0.01f ? atcg::BSDFComponentType::IdealTransmission
                                         : atcg::BSDFComponentType::GlossyTransmission;
    }

    result.bsdf_value = reflectance_color * value;

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
ATCG_HOST_DEVICE ATCG_FORCE_INLINE atcg::BSDFSamplingResult
sampleRefractive(const atcg::SurfaceInteraction& si,
                 const atcg::SampledSpectrum& reflectance_color,
                 const float roughness,
                 const float ior,
                 atcg::PCG32& rng,
                 const atcg::BSDFContext& ctx)
{
    atcg::BSDFSamplingResult result;

    glm::vec3 wi = -si.incoming_direction;    // Points towards the last traced vertex (might be the camera)

    float cos_theta_i = glm::dot(si.normal, wi);

    // Make normal point into same hemisphere as incoming direction
    glm::vec3 normal = cos_theta_i > 0.0f ? si.normal : -si.normal;

    float eta = cos_theta_i > 0.0f ? 1.0f / ior : ior;

    atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_GGX> strategy(roughness);
    glm::vec3 wh            = strategy.sample(rng.next2d());
    atcg::Frame local_frame = atcg::Frame(normal);
    wh                      = local_frame.toWorld(wh);    // Points into same hemisphere as normal

    float F = atcg::fresnel(eta, glm::dot(wi, wh));

    glm::vec3 wo;
    if(rng.next1d() < F)
    {
        wo = glm::reflect(-wi, wh);
        if(glm::dot(wo, normal) <= 0.0f)    // Reflects into the other hemisphere -> reject
        {
            result.pdf_dw = 0.0f;
            return result;
        }
    }
    else
    {
        wo         = glm::refract(-wi, wh, eta);
        result.eta = eta;
    }

    auto eval_result   = evalRefractive(si, wo, reflectance_color, roughness, ior, ctx);
    result.bsdf_weight = eval_result.bsdf_value / (eval_result.pdf_dw);
    result.pdf_dw      = eval_result.pdf_dw;
    result.out_dir     = wo;
    result.flags       = eval_result.flags;

    return result;
}

}    // namespace detail

extern "C" __device__ atcg::BSDFSamplingResult
__direct_callable__sample_dielectricbsdf(const atcg::SurfaceInteraction& si,
                                         const atcg::SampledWavelengths& wavelengths,
                                         atcg::PCG32& rng,
                                         const atcg::BSDFContext& ctx)
{
    const atcg::DielectricBSDFData* sbt_data =
        *reinterpret_cast<const atcg::DielectricBSDFData**>(optixGetSbtDataPointer());

    atcg::SampledSpectrum reflectance_color =
        atcg::SampledSpectrum::fromRGB(sbt_data->diffuse_texture.read(si.uv), wavelengths);
    float roughness = sbt_data->roughness_texture.read(si.uv);
    roughness       = glm::max(roughness * roughness, 1e-3f);    // In the real time shaders, roughness is squared

    float ior = sbt_data->ior_texture.read(si.uv);

    return detail::sampleRefractive(si, reflectance_color, roughness, ior, rng, ctx);
}

extern "C" __device__ atcg::BSDFEvalResult
__direct_callable__eval_dielectricbsdf(const atcg::SurfaceInteraction& si,
                                       const glm::vec3& outgoing_dir,
                                       const atcg::SampledWavelengths& wavelengths,
                                       const atcg::BSDFContext& ctx)
{
    const atcg::DielectricBSDFData* sbt_data =
        *reinterpret_cast<const atcg::DielectricBSDFData**>(optixGetSbtDataPointer());

    atcg::SampledSpectrum reflectance_color =
        atcg::SampledSpectrum::fromRGB(sbt_data->diffuse_texture.read(si.uv), wavelengths);
    float roughness = sbt_data->roughness_texture.read(si.uv);
    roughness       = glm::max(roughness * roughness, 1e-3f);    // In the real time shaders, roughness is squared

    float ior = sbt_data->ior_texture.read(si.uv);

    return detail::evalRefractive(si, outgoing_dir, reflectance_color, roughness, ior, ctx);
}

extern "C" __device__ float __direct_callable__evalpdf_dielectricbsdf(const atcg::SurfaceInteraction& si,
                                                                      const glm::vec3& outgoing_dir,
                                                                      const atcg::SampledWavelengths& wavelengths,
                                                                      const atcg::BSDFContext& ctx)
{
    const atcg::DielectricBSDFData* sbt_data =
        *reinterpret_cast<const atcg::DielectricBSDFData**>(optixGetSbtDataPointer());

    atcg::SampledSpectrum reflectance_color =
        atcg::SampledSpectrum::fromRGB(sbt_data->diffuse_texture.read(si.uv), wavelengths);
    float roughness = sbt_data->roughness_texture.read(si.uv);
    roughness       = glm::max(roughness * roughness, 1e-3f);    // In the real time shaders, roughness is squared

    float ior = sbt_data->ior_texture.read(si.uv);

    return detail::evalRefractivePDF(si, outgoing_dir, reflectance_color, roughness, ior);
}