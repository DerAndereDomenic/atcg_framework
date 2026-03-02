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

    glm::vec3 local_halfway = atcg::warp_square_to_hemisphere_ggx(rng.next2d(), roughness);
    float halfway_pdf       = atcg::warp_square_to_hemisphere_ggx_pdf(local_halfway, roughness);
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
            halfway_pdf * atcg::warp_normal_to_reflected_direction_pdf(wo, halfway) * reflection_probability;

        result.sample_probability = light_dir_pdf;
        NdotL                     = glm::dot(interface_normal, wo);
        HdotL                     = glm::dot(halfway, wo);
    }
    else
    {
        wo    = transmitted_ray_dir;
        HdotL = glm::dot(halfway, wo);
        float light_dir_pdf =
            halfway_pdf * atcg::warp_normal_to_refracted_direction_pdf(HdotV, HdotL, eta) * transmission_probability;

        result.sample_probability = light_dir_pdf;
        NdotL                     = -glm::dot(interface_normal, wo);
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
    result.flags       = roughness < 0.1f
                             ? atcg::BSDFComponentType::IdealReflection | atcg::BSDFComponentType::IdealTransmission
                             : atcg::BSDFComponentType::GlossyReflection | atcg::BSDFComponentType::GlossyTransmission;

    return result;
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE atcg::BSDFEvalResult evalRefractive(const atcg::SurfaceInteraction& si,
                                                                       const glm::vec3& outgoing_dir,
                                                                       const atcg::SampledSpectrum& reflectance_color,
                                                                       const float roughness,
                                                                       const float ior)
{
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

        light_dir_pdf = D * NdotH * reflection_probability * atcg::warp_normal_to_reflected_direction_pdf(wo, halfway);

        specular_bsdf = reflectance_color * D * G * F / (4.0f * NdotV * NdotL + 1e-5f);
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

        light_dir_pdf = D * NdotH * transmission_pdf * atcg::warp_normal_to_refracted_direction_pdf(VdotH, LdotH, eta);

        specular_bsdf = reflectance_color * numerator / (denom * NdotL * NdotV + 1e-5f);
    }

    atcg::BSDFEvalResult result;
    result.bsdf_value         = specular_bsdf * glm::abs(glm::dot(si.normal, wo));
    result.sample_probability = light_dir_pdf;
    result.flags              = roughness < 0.1f
                                    ? atcg::BSDFComponentType::IdealReflection | atcg::BSDFComponentType::IdealTransmission
                                    : atcg::BSDFComponentType::GlossyReflection | atcg::BSDFComponentType::GlossyTransmission;
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

template<int N>
ATCG_DEVICE ATCG_INLINE auto compute_local_frame(const CuDiff::Dual<N, glm::vec3>& localZ)
{
    auto [x, y, z] = CuDiff::unwrap(localZ);

    float sz = (z >= 0) ? 1 : -1;
    auto a   = 1 / (sz + z);
    auto ya  = y * a;
    auto b   = x * ya;
    auto c   = x * sz;

    auto localXx = c * x * a - 1;
    auto localXy = sz * b;
    auto localXz = c;

    auto localYx = b;
    auto localYy = y * ya - sz;
    auto localYz = y;
    auto localX  = CuDiff::wrap(localXx, localXy, localXz);
    auto localY  = CuDiff::wrap(localYx, localYy, localYz);

    return thrust::make_tuple(localX, localY, localZ);
}

template<int N>
ATCG_DEVICE ATCG_INLINE CuDiff::Dual<N, glm::vec3> apply_local_frame(
    const thrust::tuple<CuDiff::Dual<N, glm::vec3>, CuDiff::Dual<N, glm::vec3>, CuDiff::Dual<N, glm::vec3>>&
        local_frame,
    const CuDiff::Dual<N, glm::vec3>& v)
{
    auto [x, y, z] = CuDiff::unwrap(v);

    return thrust::get<0>(local_frame) * x + thrust::get<1>(local_frame) * y + thrust::get<2>(local_frame) * z;
}

template<int N>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE CuDiff::Dual<N, glm::vec3>
warp_square_to_hemisphere_ggx(const glm::vec2& uv, CuDiff::Dual<N, float> roughness)
{
    // GGX NDF sampling
    auto cos_theta = CuDiff::sqrt(CuDiff::max(1e-3f, (1.0f - uv.x) / (1.0f + (roughness * roughness - 1.0f) * uv.x)));
    auto sin_theta = CuDiff::sqrt(CuDiff::max(1e-3f, 1.0f - cos_theta * cos_theta));
    float phi      = 2.0f * glm::pi<float>() * uv.y;

    auto x = sin_theta * glm::cos(phi);
    auto y = sin_theta * glm::sin(phi);
    auto z = cos_theta;

    auto res = CuDiff::wrap(x, y, z);

    return res;
}

template<typename NdotHType, typename roughnessType>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto D_GGX(const NdotHType& NdotH, const roughnessType& roughness)
{
    auto a2 = roughness * roughness;
    auto d  = (NdotH * a2 - NdotH) * NdotH + 1.0f;
    return a2 / (glm::pi<float>() * d * d + 1e-5f);
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

template<typename F0Type, typename VdotHType>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto fresnel_schlick(const F0Type& F0, const VdotHType& VdotH)
{
    if constexpr(std::is_floating_point_v<VdotHType>)
    {
        return F0 + (glm::vec3(1.0f) - F0) * glm::pow(glm::max(0.0f, 1.0f - VdotH), 5.0f);
    }
    else
    {
        return F0 + (glm::vec3(1.0f) - F0) * CuDiff::pow(CuDiff::max(0.0f, 1.0f - VdotH), 5.0f);
    }
}

template<int N>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE CuDiff::Dual<N, float>
warp_normal_to_reflected_direction_pdf(const CuDiff::Dual<N, glm::vec3>& reflected_dir,
                                       const CuDiff::Dual<N, glm::vec3>& normal)
{
    return 1.0f / CuDiff::abs(4.0f * CuDiff::dot(reflected_dir, normal));
}

template<typename HdotVType, typename HdotLType, typename etaType>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto
warp_normal_to_refracted_direction_pdf(const HdotVType HdotV, const HdotLType HdotL, const etaType eta)
{
    auto denom = (HdotL + eta * HdotV);
    return eta * eta * CuDiff::abs(HdotV) / (denom * denom);
}


template<typename resultType, typename roughnessType>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto warp_square_to_hemisphere_ggx_pdf(const resultType& result,
                                                                          roughnessType roughness)
{
    auto [rx, ry, rz] = CuDiff::unwrap(result);
    return D_GGX(rz, roughness) * CuDiff::max(0.0f, rz);
}

template<typename NdotLType, typename NdotVType, typename roughnessType>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE auto G_SmithJointGGX(NdotLType NdotL, NdotVType NdotV, roughnessType roughness)
{
    auto a2      = roughness * roughness;
    auto LambdaL = 0.5f * (-1.0f + CuDiff::sqrt(1.0f + a2 * (1.0f - NdotL * NdotL) / (NdotL * NdotL)));
    auto LambdaV = 0.5f * (-1.0f + CuDiff::sqrt(1.0f + a2 * (1.0f - NdotV * NdotV) / (NdotV * NdotV)));
    return 1.0f / (1.0f + LambdaL + LambdaV);
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

    auto local_frame = compute_local_frame(interface_normal);

    auto local_halfway = warp_square_to_hemisphere_ggx(rng.next2d(), roughness);
    // Transform local halfway vector from tangent space to world space
    auto halfway     = apply_local_frame(local_frame, local_halfway);
    auto halfway_pdf = warp_square_to_hemisphere_ggx_pdf(local_halfway, roughness);

    // Compute outgoing ray directions
    auto transmitted_ray_dir = CuDiff::refract(-wi, halfway, eta);
    auto reflected_ray_dir   = CuDiff::reflect(-wi, halfway);

    // Fresnel reflectance at normal incidence
    auto F0 = (eta - 1.0f) / (eta + 1.0f);
    F0      = F0 * F0;

    // Reflection an transmission probabilities
    auto HdotV                    = CuDiff::dot(wi, halfway);
    auto F                        = fresnel_schlick(F0, HdotV);
    auto reflection_probability   = CuDiff::dot(F, glm::vec3(1.0f / 3.0f));
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
        wo                 = reflected_ray_dir;
        auto light_dir_pdf = halfway_pdf * warp_normal_to_reflected_direction_pdf(wo, halfway) * reflection_probability;

        result.sample_probability = light_dir_pdf;
        NdotL                     = CuDiff::dot(interface_normal, wo);
        HdotL                     = CuDiff::dot(halfway, wo);
    }
    else
    {
        wo    = transmitted_ray_dir;
        HdotL = CuDiff::dot(halfway, wo);
        auto light_dir_pdf =
            halfway_pdf * warp_normal_to_refracted_direction_pdf(HdotV, HdotL, eta) * transmission_probability;

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

    auto G             = G_SmithJointGGX(NdotL, NdotV, roughness);
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

#define DERIVATIVE_INTERPOLATION_VECTOR(inp_grad, texture)                                                             \
    glm::vec2 uv = texture.clamp_uv(si.uv);                                                                            \
    float fx     = uv.x * (texture.getSpecification().width - 1);                                                      \
    float fy     = uv.y * (texture.getSpecification().height - 1);                                                     \
                                                                                                                       \
    int x0 = static_cast<int>(glm::floor(fx));                                                                         \
    int y0 = static_cast<int>(glm::floor(fy));                                                                         \
    int x1 = glm::min(x0 + 1, (int)texture.getSpecification().width - 1);                                              \
    int y1 = glm::min(y0 + 1, (int)texture.getSpecification().height - 1);                                             \
                                                                                                                       \
    float tx                 = fx - x0;                                                                                \
    float ty                 = fy - y0;                                                                                \
    glm::vec3 inp_grad##_c00 = inp_grad * (1.0f - ty) * (1.0f - tx);                                                   \
    glm::vec3 inp_grad##_c10 = inp_grad * (1.0f - ty) * (tx);                                                          \
    glm::vec3 inp_grad##_c01 = inp_grad * (ty) * (1.0f - tx);                                                          \
    glm::vec3 inp_grad##_c11 = inp_grad * (ty) * (tx);                                                                 \
    float* inp_grad##_c00_p  = (float*)texture.getTexelPtr(glm::ivec2(x0, y0));                                        \
    float* inp_grad##_c10_p  = (float*)texture.getTexelPtr(glm::ivec2(x1, y0));                                        \
    float* inp_grad##_c01_p  = (float*)texture.getTexelPtr(glm::ivec2(x0, y1));                                        \
    float* inp_grad##_c11_p  = (float*)texture.getTexelPtr(glm::ivec2(x1, y1));                                        \
    atomicAdd(inp_grad##_c00_p + 0, inp_grad##_c00.x);                                                                 \
    atomicAdd(inp_grad##_c00_p + 1, inp_grad##_c00.y);                                                                 \
    atomicAdd(inp_grad##_c00_p + 2, inp_grad##_c00.z);                                                                 \
    atomicAdd(inp_grad##_c10_p + 0, inp_grad##_c10.x);                                                                 \
    atomicAdd(inp_grad##_c10_p + 1, inp_grad##_c10.y);                                                                 \
    atomicAdd(inp_grad##_c10_p + 2, inp_grad##_c10.z);                                                                 \
    atomicAdd(inp_grad##_c01_p + 0, inp_grad##_c01.x);                                                                 \
    atomicAdd(inp_grad##_c01_p + 1, inp_grad##_c01.y);                                                                 \
    atomicAdd(inp_grad##_c01_p + 2, inp_grad##_c01.z);                                                                 \
    atomicAdd(inp_grad##_c11_p + 0, inp_grad##_c11.x);                                                                 \
    atomicAdd(inp_grad##_c11_p + 1, inp_grad##_c11.y);                                                                 \
    atomicAdd(inp_grad##_c11_p + 2, inp_grad##_c11.z)

#define DERIVATIVE_INTERPOLATION_SCALAR(inp_grad, texture)                                                             \
    glm::vec2 uv = texture.clamp_uv(si.uv);                                                                            \
    float fx     = uv.x * (texture.getSpecification().width - 1);                                                      \
    float fy     = uv.y * (texture.getSpecification().height - 1);                                                     \
                                                                                                                       \
    int x0 = static_cast<int>(glm::floor(fx));                                                                         \
    int y0 = static_cast<int>(glm::floor(fy));                                                                         \
    int x1 = glm::min(x0 + 1, (int)texture.getSpecification().width - 1);                                              \
    int y1 = glm::min(y0 + 1, (int)texture.getSpecification().height - 1);                                             \
                                                                                                                       \
    float tx                = fx - x0;                                                                                 \
    float ty                = fy - y0;                                                                                 \
    float inp_grad##_c00    = inp_grad * (1.0f - ty) * (1.0f - tx);                                                    \
    float inp_grad##_c10    = inp_grad * (1.0f - ty) * (tx);                                                           \
    float inp_grad##_c01    = inp_grad * (ty) * (1.0f - tx);                                                           \
    float inp_grad##_c11    = inp_grad * (ty) * (tx);                                                                  \
    float* inp_grad##_c00_p = (float*)texture.getTexelPtr(glm::ivec2(x0, y0));                                         \
    float* inp_grad##_c10_p = (float*)texture.getTexelPtr(glm::ivec2(x1, y0));                                         \
    float* inp_grad##_c01_p = (float*)texture.getTexelPtr(glm::ivec2(x0, y1));                                         \
    float* inp_grad##_c11_p = (float*)texture.getTexelPtr(glm::ivec2(x1, y1));                                         \
    atomicAdd(inp_grad##_c00_p + 0, inp_grad##_c00);                                                                   \
    atomicAdd(inp_grad##_c10_p + 0, inp_grad##_c10);                                                                   \
    atomicAdd(inp_grad##_c01_p + 0, inp_grad##_c01);                                                                   \
    atomicAdd(inp_grad##_c11_p + 0, inp_grad##_c11)

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
    const atcg::DielectricBSDFData* sbt_data =
        *reinterpret_cast<const atcg::DielectricBSDFData**>(optixGetSbtDataPointer());

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

    glm::mat3 local_frame = atcg::Math::compute_local_frame(interface_normal);

    auto local_halfway = warp_square_to_hemisphere_ggx(rng.next2d(), roughness);
    auto halfway_pdf   = warp_square_to_hemisphere_ggx_pdf(local_halfway, roughness);
    // Transform local halfway vector from tangent space to world space
    auto halfway = local_frame * local_halfway;

    // Compute outgoing ray directions
    auto transmitted_ray_dir = CuDiff::refract(-wi, halfway, eta);
    auto reflected_ray_dir   = CuDiff::reflect(-wi, halfway);

    // Fresnel reflectance at normal incidence
    auto F0 = (eta - 1.0f) / (eta + 1.0f);
    F0      = F0 * F0;

    // Reflection an transmission probabilities
    auto HdotV                    = CuDiff::dot(wi, halfway);
    auto F                        = fresnel_schlick(F0, HdotV);
    auto reflection_probability   = CuDiff::dot(F, glm::vec3(1.0f / 3.0f));
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
        wo                 = reflected_ray_dir;
        auto light_dir_pdf = halfway_pdf * warp_normal_to_reflected_direction_pdf(wo, halfway) * reflection_probability;

        NdotL = CuDiff::dot(interface_normal, wo);
        HdotL = CuDiff::dot(halfway, wo);
    }
    else
    {
        wo    = transmitted_ray_dir;
        HdotL = CuDiff::dot(halfway, wo);
        auto light_dir_pdf =
            halfway_pdf * warp_normal_to_refracted_direction_pdf(HdotV, HdotL, eta) * transmission_probability;

        NdotL = -CuDiff::dot(interface_normal, wo);
    }

    if(NdotL <= 0)
    {
        return;
    }

    float NdotV = glm::abs(glm::dot(interface_normal, wi));
    auto NdotH  = CuDiff::dot(halfway, interface_normal);

    auto G           = G_SmithJointGGX(NdotL, NdotV, roughness);
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
        DERIVATIVE_INTERPOLATION_VECTOR(dLdalbedo, sbt_data->diffuse_grad);
    }
    {
        DERIVATIVE_INTERPOLATION_SCALAR(dLdior, sbt_data->ior_grad);
    }
    {
        DERIVATIVE_INTERPOLATION_SCALAR(dLdr, sbt_data->roughness_grad);
    }
}