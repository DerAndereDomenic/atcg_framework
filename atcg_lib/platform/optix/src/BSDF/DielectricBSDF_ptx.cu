#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>

#include <Math/Functions.h>
#include <Core/SurfaceInteraction.h>
#include <BSDF/BSDFVPtrTable.cuh>
#include <BSDF/DielectricBSDFData.cuh>
#include <BSDF/BSDFFunctions.h>
#include <BSDF/Sampling.h>

#include <CuDiff/ext/glm.h>
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

extern "C" __device__ atcg::BSDFDualSamplingResult
__direct_callable__sample_forward_dielectricbsdf(const atcg::DualSurfaceInteraction& si, atcg::PCG32& rng)
{
    const atcg::DielectricBSDFData* sbt_data =
        *reinterpret_cast<const atcg::DielectricBSDFData**>(optixGetSbtDataPointer());

    auto reflectance_color = sbt_data->diffuse_texture.read(si.uv);
    auto roughness         = sbt_data->roughness_texture.read(si.uv);
    roughness = CuDiff::max(roughness * roughness, 1e-3f);    // In the real time shaders, roughness is squared

    auto ior = sbt_data->ior_texture.read(si.uv);

    auto wi = -si.incoming_direction;

    // Determine surface parameters
    bool outsidein        = glm::dot(wi.val(), si.normal.val()) > 0;
    auto interface_normal = outsidein ? si.normal : -si.normal;
    auto eta              = outsidein ? 1.0f / ior : ior;

    auto local_frame = atcg::Frame(interface_normal);

    atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_GGX, decltype(roughness)> strategy(roughness);
    auto local_halfway = strategy.sample(rng.next2d());
    // Transform local halfway vector from tangent space to world space
    auto halfway     = local_frame.toWorld(local_halfway);
    auto halfway_pdf = strategy.pdf(local_halfway);

    // Compute outgoing ray directions
    auto transmitted_ray_dir = CuDiff::refract(-wi, halfway, eta);
    auto reflected_ray_dir   = CuDiff::reflect(-wi, halfway);

    // Fresnel reflectance at normal incidence
    auto F0 = (eta - 1.0f) / (eta + 1.0f);
    F0      = F0 * F0;

    // Reflection an transmission probabilities
    auto HdotV                    = CuDiff::dot(wi, halfway);
    auto F                        = atcg::fresnel_schlick(F0, HdotV);
    auto reflection_probability   = F;
    auto transmission_probability = 1.0f - reflection_probability;
    if(glm::dot(transmitted_ray_dir.val(), transmitted_ray_dir.val()) < 1e-6f)
    {
        // Total internal reflection!
        transmission_probability = CuDiff::Dual<6, float>(0.0f);
        reflection_probability   = CuDiff::Dual<6, float>(1.0f);
    }

    // Compute sampling result
    atcg::BSDFDualSamplingResult result;
    result.sample_probability = CuDiff::Dual<6, float>(0);

    // Stochastically select a reflection or transmission via russian roulette
    CuDiff::Dual<6, glm::vec3> wo;
    CuDiff::Dual<6, float> NdotL;
    CuDiff::Dual<6, float> HdotL;
    if(rng.next1d() < reflection_probability)
    {
        wo = reflected_ray_dir;
        auto light_dir_pdf =
            halfway_pdf *
            atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_GGX>::warp_halfway_to_reflected_direction_pdf(
                wo,
                halfway) *
            reflection_probability;

        result.sample_probability = light_dir_pdf;
        NdotL                     = CuDiff::dot(interface_normal, wo);
        HdotL                     = CuDiff::dot(halfway, wo);
    }
    else
    {
        wo    = transmitted_ray_dir;
        HdotL = CuDiff::dot(halfway, wo);
        auto light_dir_pdf =
            halfway_pdf *
            atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_GGX>::warp_halfway_to_refracted_direction_pdf(
                HdotV,
                HdotL,
                eta) *
            transmission_probability;

        result.sample_probability = light_dir_pdf;
        NdotL                     = -CuDiff::dot(interface_normal, wo);
    }

    if(NdotL <= 0)
    {
        result.sample_probability = CuDiff::Dual<6, float>(0.0f);
        return result;
    }

    auto NdotV = CuDiff::abs(CuDiff::dot(interface_normal, wi));
    auto NdotH = CuDiff::dot(halfway, interface_normal);

    auto G             = atcg::G_SmithJointGGX(NdotL, NdotV, roughness);
    result.bsdf_weight = reflectance_color * G * CuDiff::abs(HdotL) / (NdotV * NdotH);
    result.out_dir     = wo;
    result.flags       = roughness < 0.1f
                             ? atcg::BSDFComponentType::IdealReflection | atcg::BSDFComponentType::IdealReflection
                             : atcg::BSDFComponentType::GlossyReflection | atcg::BSDFComponentType::GlossyTransmission;

    return result;
}

extern "C" __device__ atcg::BSDFDualEvalResult
__direct_callable__eval_forward_dielectricbsdf(const atcg::SurfaceInteraction& si, const glm::vec3& outgoing_dir)
{
    return atcg::BSDFDualEvalResult();
}

extern "C" __device__ void __direct_callable__eval_backward_dielectricbsdf(const atcg::SurfaceInteraction& si,
                                                                           const glm::vec3& outgoing_dir,
                                                                           const glm::vec3& dLdbsdf)
{
}

extern "C" __device__ void __direct_callable__sample_backward_dielectricbsdf(const atcg::SurfaceInteraction& si,
                                                                             atcg::PCG32& rng,
                                                                             const glm::vec3& dLdbsdf,
                                                                             const glm::vec2& dLdwo)
{
    atcg::DielectricBSDFData* sbt_data = *reinterpret_cast<atcg::DielectricBSDFData**>(optixGetSbtDataPointer());

    if(!sbt_data->optimizable) return;

    if(isnan(dLdwo.x) || isnan(dLdwo.y)) return;

    glm::vec3 reflectance_color_ = sbt_data->diffuse_texture.read(si.uv);
    float roughness_             = sbt_data->roughness_texture.read(si.uv);
    float ior_                   = sbt_data->ior_texture.read(si.uv);

    auto [reflectance_color, r, ior] = CuDiff::make_variables<5>(reflectance_color_, roughness_, ior_);

    auto roughness = r * r;    // In the real time shaders, roughness is squared
    if(roughness.val() < 1e-3f) roughness.mut_val() = 1e-3f;

    glm::vec3 wi = -si.incoming_direction;

    // Determine surface parameters
    bool outsidein             = glm::dot(wi, si.normal) > 0;
    glm::vec3 interface_normal = outsidein ? si.normal : -si.normal;
    auto eta                   = outsidein ? 1.0f / ior : ior;

    atcg::Frame<glm::vec3> local_frame = atcg::Frame(interface_normal);

    atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_GGX, decltype(roughness)> strategy(roughness);
    auto local_halfway = strategy.sample(rng.next2d());
    auto halfway_pdf   = strategy.pdf(local_halfway);
    // Transform local halfway vector from tangent space to world space
    auto halfway = local_frame.toWorld(local_halfway);

    // Compute outgoing ray directions
    auto transmitted_ray_dir = CuDiff::refract(-wi, halfway, eta);
    auto reflected_ray_dir   = CuDiff::reflect(-wi, halfway);

    // Fresnel reflectance at normal incidence
    auto F0 = (eta - 1.0f) / (eta + 1.0f);
    F0      = F0 * F0;

    // Reflection an transmission probabilities
    auto HdotV                    = CuDiff::dot(wi, halfway);
    auto F                        = atcg::fresnel_schlick(F0, HdotV);
    auto reflection_probability   = F;
    auto transmission_probability = 1.0f - reflection_probability;
    if(glm::dot(transmitted_ray_dir.val(), transmitted_ray_dir.val()) < 1e-6f)
    {
        // Total internal reflection!
        transmission_probability = CuDiff::Dual<5, float>(0.0f);
        reflection_probability   = CuDiff::Dual<5, float>(1.0f);
    }

    // Stochastically select a reflection or transmission via russian roulette
    CuDiff::Dual<5, glm::vec3> wo;
    CuDiff::Dual<5, float> NdotL;
    CuDiff::Dual<5, float> HdotL;
    if(rng.next1d() < reflection_probability)
    {
        wo = reflected_ray_dir;
        auto light_dir_pdf =
            halfway_pdf *
            atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_GGX>::warp_halfway_to_reflected_direction_pdf(
                wo,
                halfway) *
            reflection_probability;

        NdotL = CuDiff::dot(interface_normal, wo);
        HdotL = CuDiff::dot(halfway, wo);
    }
    else
    {
        wo    = transmitted_ray_dir;
        HdotL = CuDiff::dot(halfway, wo);
        auto light_dir_pdf =
            halfway_pdf *
            atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_GGX>::warp_halfway_to_refracted_direction_pdf(
                HdotV,
                HdotL,
                eta) *
            transmission_probability;

        NdotL = -CuDiff::dot(interface_normal, wo);
    }

    if(NdotL <= 0)
    {
        return;
    }

    float NdotV = glm::abs(glm::dot(interface_normal, wi));
    auto NdotH  = CuDiff::dot(halfway, interface_normal);

    auto G           = atcg::G_SmithJointGGX(NdotL, NdotV, roughness);
    auto bsdf_weight = reflectance_color * G * CuDiff::abs(HdotL) / (NdotV * NdotH);
    auto out_dir     = wo;

    auto [dx, dy, dz] = CuDiff::unwrap(out_dir);
    auto dy_clamp     = CuDiff::clamp(dy, -1.0f, 1.0f);
    auto theta_n      = CuDiff::acos(dy_clamp);
    auto phi_n        = CuDiff::atan2(dz, dx);

    glm::mat3x2 dwodalbedo = glm::mat3x2(glm::vec2(phi_n.derivative(0), theta_n.derivative(0)),
                                         glm::vec2(phi_n.derivative(1), theta_n.derivative(1)),
                                         glm::vec2(phi_n.derivative(2), theta_n.derivative(2)));
    glm::vec2 dwodr        = glm::vec2(phi_n.derivative(3), theta_n.derivative(3));
    glm::vec2 dwodior      = glm::vec2(phi_n.derivative(4), theta_n.derivative(4));

    glm::mat3 dbsdf_weightdalbedo =
        glm::mat3(bsdf_weight.derivative(0), bsdf_weight.derivative(1), bsdf_weight.derivative(2));
    glm::vec3 dbsdf_weightdr   = bsdf_weight.derivative(3);
    glm::vec3 dbsdf_weightdior = bsdf_weight.derivative(4);

    glm::vec3 dLdalbedo = dLdbsdf * dbsdf_weightdalbedo + dLdwo * dwodalbedo;
    float dLdr          = glm::dot(dLdbsdf, dbsdf_weightdr) + glm::dot(dLdwo, dwodr);
    float dLdior        = glm::dot(dLdbsdf, dbsdf_weightdior) + glm::dot(dLdwo, dwodior);

    if(isnan(glm::length2(dLdalbedo)) || isnan(dLdr) || isnan(dLdr)) return;

    {
        sbt_data->diffuse_grad.write<glm::vec2, atcg::TexelWriteMode::ATOMIC_ADD>(dLdalbedo, si.uv);
    }
    {
        sbt_data->ior_grad.write<glm::vec2, atcg::TexelWriteMode::ATOMIC_ADD>(dLdior, si.uv);
    }
    {
        sbt_data->roughness_grad.write<glm::vec2, atcg::TexelWriteMode::ATOMIC_ADD>(dLdr, si.uv);
    }
}