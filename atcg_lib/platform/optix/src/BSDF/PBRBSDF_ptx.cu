#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>

#include <Math/Functions.h>
#include <Core/SurfaceInteraction.h>
#include <BSDF/BSDFVPtrTable.cuh>
#include <BSDF/PBRBSDFData.cuh>

#include <CuDiff/CuDiff.h>
#include <CuDiff/ext/glm.h>

namespace detail
{

ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec3 dfresnel_schlickdM(const float VdotH)
{
    return glm::vec3(1.0f) - glm::pow(glm::max(0.0f, 1.0f - VdotH), 5.0f);
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE float dD_GGXdR(const float NdotH, const float roughness)
{
    float r      = roughness;
    float r2     = r * r;
    float NdotH2 = NdotH * NdotH;
    float num    = 2.0f * r * (-2.0f * NdotH2 * r2 + NdotH2 * (r2 - 1.0f) + 1.0f);
    float denom  = (NdotH2 * (r2 - 1.0f) + 1.0f);

    return num / (glm::pi<float>() * denom * denom * denom + 1e-5f);
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T dV_SmithGGXdR(T NdotL, T NdotV, T alpha, T eps = 1e-8f)
{
    T r      = alpha;
    T r2     = r * r;
    T NdotL2 = NdotL * NdotL;
    T NdotV2 = NdotV * NdotV;

    T p1       = glm::sqrt(-NdotL2 * (r2 - T(1)) + r2);
    T p2       = glm::sqrt(-NdotV2 * (r2 - T(1)) + r2);
    T dLambdaV = NdotL * p1;
    T dLambdaL = NdotV * p2;

    T num   = T(0.5) * r * (dLambdaL * (NdotL2 - T(1)) + dLambdaV * (NdotV2 - T(1)));
    T denom = NdotL * p2 + NdotV * p1;
    denom   = denom * denom;
    denom   = denom * p1 * p2;

    return num / (denom + eps);
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
ATCG_HOST_DEVICE ATCG_FORCE_INLINE atcg::BSDFSamplingResult samplePBR(const atcg::SurfaceInteraction& si,
                                                                      const glm::vec3& diffuse_color,
                                                                      const glm::vec3& specular_F0,
                                                                      const float& metallic,
                                                                      const float& roughness,
                                                                      atcg::PCG32& rng)
{
    atcg::BSDFSamplingResult result;

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
    glm::mat3 local_frame = atcg::Math::compute_local_frame(normal);

    float diffuse_probability = glm::dot(diffuse_color, glm::vec3(1)) /
                                (glm::dot(diffuse_color, glm::vec3(1)) + glm::dot(specular_F0, glm::vec3(1)) + 1e-5f);
    float specular_probability = 1 - diffuse_probability;

    if(rng.next1d() < diffuse_probability)
    {
        // Sample light direction from diffuse bsdf
        glm::vec3 local_outgoing_ray_dir = atcg::warp_square_to_hemisphere_cosine(rng.next2d());
        // Transform local outgoing direction from tangent space to world space
        result.out_dir = local_frame * local_outgoing_ray_dir;
    }
    else
    {
        // Sample light direction from specular bsdf
        glm::vec3 local_halfway = atcg::warp_square_to_hemisphere_ggx(rng.next2d(), roughness);
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
        float NDF = atcg::D_GGX(NdotH, roughness);

        // Visibility
        float V = atcg::V_SmithGGX(NdotL, NdotV, roughness);

        // Fresnel
        glm::vec3 F = atcg::fresnel_schlick(specular_F0, HdotV);

        kD = (1.0f - F);

        specular_bsdf = NDF * V * F;

        float halfway_pdf = NDF * NdotH;
        float halfway_to_outgoing_pdf =
            atcg::warp_normal_to_reflected_direction_pdf(result.out_dir, halfway);    // 1 / (4*HdotV)
        specular_pdf = halfway_pdf * halfway_to_outgoing_pdf;
    }

    result.sample_probability = diffuse_probability * diffuse_pdf + specular_probability * specular_pdf;
    result.bsdf_weight        = (specular_bsdf + kD * diffuse_bsdf) * NdotL / (result.sample_probability + 1e-5f);
    result.flags =
        result.flags | (roughness < 0.1f ? atcg::BSDFComponentType::IdealReflection : atcg::BSDFComponentType::Any);

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
ATCG_HOST_DEVICE ATCG_FORCE_INLINE atcg::BSDFEvalResult evalPBR(const atcg::SurfaceInteraction& si,
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
    float V     = atcg::V_SmithGGX(NdotL, NdotV, roughness);
    glm::vec3 F = atcg::fresnel_schlick(metallic_color, glm::max(glm::dot(H, view_dir), 0.0f));

    glm::vec3 specular = NDF * V * F;

    glm::vec3 kS = F;
    glm::vec3 kD = glm::vec3(1.0) - kS;

    float diffuse_probability =
        glm::dot(diffuse_color, glm::vec3(1)) /
        (glm::dot(diffuse_color, glm::vec3(1)) + glm::dot(metallic_color, glm::vec3(1)) + 1e-5f);
    float specular_probability    = 1 - diffuse_probability;
    float diffuse_pdf             = NdotL / glm::pi<float>();
    float halfway_pdf             = NDF * NdotH;
    float halfway_to_outgoing_pdf = atcg::warp_normal_to_reflected_direction_pdf(outgoing_dir, H);    // 1 / (4*HdotV)
    float specular_pdf            = halfway_pdf * halfway_to_outgoing_pdf;

    result.bsdf_value         = specular + kD * diffuse_color / glm::pi<float>();
    result.sample_probability = diffuse_probability * diffuse_pdf + specular_probability * specular_pdf;
    result.flags =
        result.flags | (roughness < 0.1f ? atcg::BSDFComponentType::IdealReflection : atcg::BSDFComponentType::Any);

    return result;
}
}    // namespace detail

extern "C" __device__ atcg::BSDFSamplingResult __direct_callable__sample_pbrbsdf(const atcg::SurfaceInteraction& si,
                                                                                 atcg::PCG32& rng)
{
    const atcg::PBRBSDFData* sbt_data = *reinterpret_cast<const atcg::PBRBSDFData**>(optixGetSbtDataPointer());

    glm::vec3 diffuse_color = sbt_data->diffuse_texture.read(si.uv);
    float metallic          = sbt_data->metallic_texture.read(si.uv);
    float roughness         = sbt_data->roughness_texture.read(si.uv);
    roughness = glm::max(roughness * roughness, 1e-3f);    // In the real time shaders, roughness is squared

    glm::vec3 metallic_color = (1.0f - metallic) * glm::vec3(0.04f) + metallic * diffuse_color;
    diffuse_color            = glm::lerp(diffuse_color, glm::vec3(0), metallic) * si.color;

    return detail::samplePBR(si, diffuse_color, metallic_color, metallic, roughness, rng);
}

extern "C" __device__ atcg::BSDFEvalResult __direct_callable__eval_pbrbsdf(const atcg::SurfaceInteraction& si,
                                                                           const glm::vec3& outgoing_dir)
{
    const atcg::PBRBSDFData* sbt_data = *reinterpret_cast<const atcg::PBRBSDFData**>(optixGetSbtDataPointer());
    atcg::BSDFEvalResult result;

    glm::vec3 diffuse_color = sbt_data->diffuse_texture.read(si.uv);
    float metallic          = sbt_data->metallic_texture.read(si.uv);
    float roughness         = sbt_data->roughness_texture.read(si.uv);
    roughness = glm::max(roughness * roughness, 1e-3f);    // In the real time shaders, roughness is squared
    glm::vec3 metallic_color = (1.0f - metallic) * glm::vec3(0.04f) + metallic * diffuse_color;
    diffuse_color            = glm::lerp(diffuse_color, glm::vec3(0), metallic) * si.color;


    return detail::evalPBR(si, outgoing_dir, diffuse_color, metallic_color, roughness, metallic);
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

ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec3 warp_square_to_hemisphere_ggx_derivative(const glm::vec2& uv,
                                                                                      const float roughness)
{
    float u      = uv.x;
    float v      = uv.y;
    float alpha  = roughness;
    float alpha2 = alpha * alpha;
    float phi    = glm::two_pi<float>() * v;

    float a  = alpha * u * (u - 1.0f);
    float b  = u * (alpha2 - 1.0f) + 1.0f;
    float b2 = b * b;
    float c  = glm::sqrt((alpha2 * u) / b);

    float x = -(a * glm::cos(phi)) / (c * b2);
    float y = -(a * glm::sin(phi)) / (c * b2);
    float z = -(alpha * u * glm::sqrt(-(u - 1.0f) / b)) / b;

    return glm::vec3(x, y, z);
}

extern "C" __device__ atcg::BSDFDualSamplingResult
__direct_callable__sample_forward_pbrbsdf(const atcg::DualSurfaceInteraction& si, atcg::PCG32& rng)
{
    const atcg::PBRBSDFData* sbt_data = *reinterpret_cast<const atcg::PBRBSDFData**>(optixGetSbtDataPointer());

    auto diffuse_color = sbt_data->diffuse_texture.read(si.uv);
    auto metallic      = sbt_data->metallic_texture.read(si.uv);
    auto roughness     = sbt_data->roughness_texture.read(si.uv);
    // if(roughness.val() < 1e-3f)
    // {
    //     roughness.mut_val() = 1e-3f;
    // }
    // roughness = roughness * roughness;
    roughness = CuDiff::max(roughness * roughness, 1e-3f);    // TODO: We only clamp values but not
    // derivative to not loose them. Correct?

    auto metallic_color = (1.0f - metallic) * glm::vec3(0.04f) + metallic * diffuse_color;
    diffuse_color       = (1.0f - metallic) * diffuse_color;


    // Do the sampling
    atcg::BSDFDualSamplingResult result;

    // Direction towards viewer
    auto view_dir = -si.incoming_direction;
    auto normal   = si.normal;


    // Don't trace a new ray if surface is viewed from below
    auto NdotV = CuDiff::dot(normal, view_dir);
    if(NdotV <= 0)
    {
        return result;
    }

    // The matrix local_frame transforms a vector from the coordinate system where geom.N corresponds to the z-axis to
    // the world coordinate system.
    auto local_frame = compute_local_frame(normal);

    auto diffuse_probability =
        CuDiff::dot(diffuse_color, glm::vec3(1)) /
        (CuDiff::dot(diffuse_color, glm::vec3(1)) + CuDiff::dot(metallic_color, glm::vec3(1)) + 1e-5f);
    auto specular_probability = 1.0f - diffuse_probability;
    if(rng.next1d() < diffuse_probability)
    {
        // Sample light direction from diffuse bsdf
        glm::vec3 local_outgoing_ray_dir = atcg::warp_square_to_hemisphere_cosine(rng.next2d());
        // Transform local outgoing direction from tangent space to world space
        result.out_dir = apply_local_frame(local_frame, CuDiff::Dual<4, glm::vec3>(local_outgoing_ray_dir));
    }
    else
    {
        // Sample light direction from specular bsdf
        auto local_halfway = warp_square_to_hemisphere_ggx(rng.next2d(), roughness);
        // Transform local halfway vector from tangent space to world space
        auto halfway   = apply_local_frame(local_frame, local_halfway);
        result.out_dir = CuDiff::reflect(si.incoming_direction, halfway);
    }

    // It is possible that light directions below the horizon are sampled..
    // If outgoing ray direction is below horizon, let the sampling fail!
    auto NdotL = CuDiff::dot(normal, result.out_dir);
    if(NdotL <= 0)
    {
        result.sample_probability = 0;
        return result;
    }

    auto diffuse_bsdf = diffuse_color / glm::pi<float>();
    auto diffuse_pdf  = NdotL / glm::pi<float>();

    CuDiff::Dual<4, glm::vec3> specular_bsdf = glm::vec3(0);
    CuDiff::Dual<4, float> specular_pdf      = 0.0f;
    // Only compute specular component if specular_f0 is not zero!
    CuDiff::Dual<4, glm::vec3> kD = glm::vec3(1);
    if(CuDiff::dot(metallic_color, metallic_color) > 1e-6f)
    {
        auto halfway = CuDiff::normalize(result.out_dir + view_dir);
        auto HdotV   = CuDiff::dot(halfway, result.out_dir);
        auto NdotH   = CuDiff::dot(halfway, normal);

        // Normal distribution
        auto NDF = D_GGX(NdotH, roughness);

        // Visibility
        auto V = V_SmithGGX(NdotL, NdotV, roughness);

        // Fresnel
        auto F = fresnel_schlick(metallic_color, HdotV);

        kD = (1.0f - F);

        specular_bsdf = NDF * V * F;

        auto halfway_pdf             = NDF * NdotH;
        auto halfway_to_outgoing_pdf = warp_normal_to_reflected_direction_pdf(result.out_dir, halfway);
        specular_pdf                 = halfway_pdf * halfway_to_outgoing_pdf;
    }

    result.sample_probability = diffuse_probability * diffuse_pdf + specular_probability * specular_pdf;
    result.bsdf_weight        = (specular_bsdf + kD * diffuse_bsdf) * NdotL / (result.sample_probability + 1e-5f);

    result.flags =
        result.flags | (roughness < 0.1f ? atcg::BSDFComponentType::IdealReflection : atcg::BSDFComponentType::Any);

    return result;
}

extern "C" __device__ atcg::BSDFDualEvalResult
__direct_callable__eval_forward_pbrbsdf(const atcg::SurfaceInteraction& si, const glm::vec3& outgoing_dir)
{
    // const atcg::PBRBSDFData* sbt_data = *reinterpret_cast<const atcg::PBRBSDFData**>(optixGetSbtDataPointer());
    // atcg::BSDFEvalResult result;

    // glm::vec3 diffuse_color = sbt_data->diffuse_texture.read(si.uv);
    // float metallic          = sbt_data->metallic_texture.read(si.uv);
    // float roughness         = sbt_data->roughness_texture.read(si.uv);
    // roughness = glm::max(roughness * roughness, 1e-3f);    // In the real time shaders, roughness is squared
    // glm::vec3 metallic_color = (1.0f - metallic) * glm::vec3(0.04f) + metallic * diffuse_color;
    // diffuse_color            = glm::lerp(diffuse_color, glm::vec3(0), metallic) * si.color;


    // return detail::evalPBR(si, outgoing_dir, diffuse_color, metallic_color, roughness, metallic);
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

extern "C" __device__ void __direct_callable__eval_backward_pbrbsdf(const atcg::SurfaceInteraction& si,
                                                                    const glm::vec3& outgoing_dir,
                                                                    const glm::vec3& dLdbsdf)
{
    {
        const atcg::PBRBSDFData* sbt_data = *reinterpret_cast<const atcg::PBRBSDFData**>(optixGetSbtDataPointer());

        if(!sbt_data->optimizable) return;

        glm::vec3 alpha_ = sbt_data->diffuse_texture.read(si.uv);
        float m_         = sbt_data->metallic_texture.read(si.uv);
        float r_         = sbt_data->roughness_texture.read(si.uv);

        auto [alpha, m, r] = CuDiff::make_variables<5>(alpha_, m_, r_);

        auto roughness = r * r;    // In the real time shaders, roughness is squared
        if(roughness.val() < 1e-3f) roughness.mut_val() = 1e-3f;
        auto diffuse_color = (1.0f - m) * alpha;    // glm::lerp(alpha, glm::vec3(0), m) * si.color;

        auto metallic_color = (1.0f - m) * glm::vec3(0.04f) + m * alpha;

        glm::vec3 light_dir = outgoing_dir;
        glm::vec3 view_dir  = -si.incoming_direction;

        glm::vec3 H = glm::normalize(light_dir + view_dir);

        float NdotH = glm::max(glm::dot(si.normal, H), 0.0f);
        float NdotV = glm::max(glm::dot(si.normal, view_dir), 0.0f);
        float NdotL = glm::max(glm::dot(si.normal, light_dir), 0.0f);
        float VdotH = glm::max(glm::dot(H, view_dir), 0.0f);

        if(NdotL <= 0.0f || NdotV <= 0.0f) return;

        auto NDF = D_GGX(NdotH, roughness);
        auto V   = V_SmithGGX(NdotL, NdotV, roughness);
        auto F   = fresnel_schlick(metallic_color, VdotH);
        auto kS  = F;
        auto kD  = glm::vec3(1.0) - kS;

        auto specular = NDF * V * F;

        auto bsdf_value = specular + kD * diffuse_color / glm::pi<float>();

        glm::mat3 dbsdfdalpha = glm::mat3(bsdf_value.derivative(0), bsdf_value.derivative(1), bsdf_value.derivative(2));
        glm::vec3 dbsdfdm     = bsdf_value.derivative(3);
        glm::vec3 dbsdfdr     = bsdf_value.derivative(4);

        glm::vec3 dLdalbedo = dLdbsdf * dbsdfdalpha;
        float dLdr          = glm::dot(dbsdfdr, dLdbsdf);
        float dLdm          = glm::dot(dbsdfdm, dLdbsdf);

        {
            DERIVATIVE_INTERPOLATION_VECTOR(dLdalbedo, sbt_data->diffuse_grad);
        }
        {
            DERIVATIVE_INTERPOLATION_SCALAR(dLdr, sbt_data->roughness_grad);
        }
        {
            DERIVATIVE_INTERPOLATION_SCALAR(dLdm, sbt_data->metallic_grad);
        }
    }

    // {
    //     const atcg::PBRBSDFData* sbt_data = *reinterpret_cast<const atcg::PBRBSDFData**>(optixGetSbtDataPointer());

    //     if(!sbt_data->optimizable) return;

    //     glm::vec3 alpha         = sbt_data->diffuse_texture.read(si.uv);
    //     float m                 = sbt_data->metallic_texture.read(si.uv);
    //     float r                 = sbt_data->roughness_texture.read(si.uv);
    //     float roughness         = glm::max(r * r, 1e-3f);    // In the real time shaders, roughness is squared
    //     glm::vec3 diffuse_color = glm::lerp(alpha, glm::vec3(0), m) * si.color;

    //     glm::vec3 metallic_color = (1.0f - m) * glm::vec3(0.04f) + m * alpha;

    //     glm::vec3 light_dir = outgoing_dir;
    //     glm::vec3 view_dir  = -si.incoming_direction;

    //     glm::vec3 H = glm::normalize(light_dir + view_dir);

    //     float NdotH = glm::max(glm::dot(si.normal, H), 0.0f);
    //     float NdotV = glm::max(glm::dot(si.normal, view_dir), 0.0f);
    //     float NdotL = glm::max(glm::dot(si.normal, light_dir), 0.0f);
    //     float VdotH = glm::max(glm::dot(H, view_dir), 0.0f);

    //     if(NdotL <= 0.0f || NdotV <= 0.0f) return;

    //     float NDF    = atcg::D_GGX(NdotH, roughness);
    //     float V      = atcg::V_SmithGGX(NdotL, NdotV, roughness);
    //     glm::vec3 F  = atcg::fresnel_schlick(metallic_color, VdotH);
    //     glm::vec3 kS = F;
    //     glm::vec3 kD = glm::vec3(1.0) - kS;

    //     glm::vec3 dFdM         = detail::dfresnel_schlickdM(VdotH);
    //     float dNDFdR           = detail::dD_GGXdR(NdotH, roughness);
    //     float dVdR             = detail::dV_SmithGGXdR(NdotL, NdotV, roughness);

    //     glm::vec3 diffuse_bsdf = glm::one_over_pi<float>() * diffuse_color;

    //     glm::vec3 dbsdfdalbedo =
    //         kD * glm::one_over_pi<float>() * (1.0f - m) + m * dFdM * (V * NDF - diffuse_bsdf);    // Diagonal matrix
    //     glm::vec3 dLdalbedo = dbsdfdalbedo * dLdbsdf;

    //     glm::vec3 dbsdfdroughness = 2.0f * r * (F * dNDFdR * V + F * NDF * dVdR);
    //     float dLdroughness        = glm::dot(dbsdfdroughness, dLdbsdf);

    //     glm::vec3 dbsdfdmetallic =
    //         (alpha - glm::vec3(0.04f)) * dFdM * (V * NDF - diffuse_bsdf) - alpha * kD * glm::one_over_pi<float>();
    //     float dLdmetallic = glm::dot(dbsdfdmetallic, dLdbsdf);

    //     {
    //         DERIVATIVE_INTERPOLATION_VECTOR(dLdalbedo, sbt_data->diffuse_grad);
    //     }
    //     {
    //         DERIVATIVE_INTERPOLATION_SCALAR(dLdroughness, sbt_data->roughness_grad);
    //     }
    //     {
    //         DERIVATIVE_INTERPOLATION_SCALAR(dLdmetallic, sbt_data->metallic_grad);
    //     }
    // }
}

extern "C" __device__ void __direct_callable__sample_backward_pbrbsdf(const atcg::SurfaceInteraction& si,
                                                                      atcg::PCG32& rng,
                                                                      const glm::vec3& dLdbsdf,
                                                                      const glm::vec2& dLdwo)
{
    {
        const atcg::PBRBSDFData* sbt_data = *reinterpret_cast<const atcg::PBRBSDFData**>(optixGetSbtDataPointer());

        if(!sbt_data->optimizable) return;

        if(isnan(dLdwo.x) || isnan(dLdwo.y)) return;

        glm::vec3 albedo_ = sbt_data->diffuse_texture.read(si.uv);
        float m_          = sbt_data->metallic_texture.read(si.uv);
        float r_          = sbt_data->roughness_texture.read(si.uv);

        auto [albedo, m, r] = CuDiff::make_variables<5>(albedo_, m_, r_);

        // auto roughness = CuDiff::max(r * r, 1e-3f);    // In the real time shaders, roughness is squared
        auto roughness = r * r;    // In the real time shaders, roughness is squared
        if(roughness.val() < 1e-3f) roughness.mut_val() = 1e-3f;
        auto diffuse_color = (1.0f - m) * albedo;

        auto metallic_color = (1.0f - m) * glm::vec3(0.04f) + m * albedo;

        // Direction towards viewer
        glm::vec3 view_dir = -si.incoming_direction;
        glm::vec3 normal   = si.normal;

        // Don't trace a new ray if surface is viewed from below
        float NdotV = glm::dot(normal, view_dir);
        if(NdotV <= 0)
        {
            return;
        }

        // The matrix local_frame transforms a vector from the coordinate system where geom.N corresponds to the z-axis
        // to the world coordinate system.
        glm::mat3 local_frame = atcg::Math::compute_local_frame(normal);

        auto diffuse_probability =
            CuDiff::dot(diffuse_color, glm::vec3(1)) /
            (CuDiff::dot(diffuse_color, glm::vec3(1)) + CuDiff::dot(metallic_color, glm::vec3(1)) + 1e-5f);
        auto specular_probability = 1 - diffuse_probability;

        CuDiff::Dual<5, glm::vec3> out_dir;
        if(rng.next1d() < diffuse_probability)
        {
            // Sample light direction from diffuse bsdf
            glm::vec3 local_outgoing_ray_dir = atcg::warp_square_to_hemisphere_cosine(rng.next2d());
            // Transform local outgoing direction from tangent space to world space
            out_dir = local_frame * local_outgoing_ray_dir;
        }
        else
        {
            // Sample light direction from specular bsdf
            auto local_halfway = warp_square_to_hemisphere_ggx(rng.next2d(), roughness);
            // Transform local halfway vector from tangent space to world space
            auto halfway = local_frame * local_halfway;
            out_dir      = CuDiff::reflect(si.incoming_direction, halfway);
        }

        auto [dx, dy, dz] = CuDiff::unwrap(out_dir);
        auto dy_clamp     = CuDiff::clamp(dy, -1.0f, 1.0f);
        auto theta_n      = CuDiff::acos(dy_clamp);
        auto phi_n        = CuDiff::atan2(dz, dx);

        glm::mat3x2 dwodalbedo = glm::mat3x2(glm::vec2(phi_n.derivative(0), theta_n.derivative(0)),
                                             glm::vec2(phi_n.derivative(1), theta_n.derivative(1)),
                                             glm::vec2(phi_n.derivative(2), theta_n.derivative(2)));
        glm::vec2 dwodm        = glm::vec2(phi_n.derivative(3), theta_n.derivative(3));
        glm::vec2 dwodr        = glm::vec2(phi_n.derivative(4), theta_n.derivative(4));

        // I think that light_dir needs to be detached here because of these lines in the pseudo code:
        // # Backpropagate gradients of the current BSDF value
        // δπ += backward_grad(bsdf_weight, δL ∗ L / bsdf_weight)
        // # Backpropagate through shading frame and
        // # BSDF sampling calculation
        // δπ += backward_grad(ray′, δL @ J′  L)
        CuDiff::Dual<5, glm::vec3> light_dir = out_dir;

        // It is possible that light directions below the horizon are sampled..
        // If outgoing ray direction is below horizon, let the sampling fail!
        auto NdotL = CuDiff::dot(normal, light_dir);
        if(NdotL <= 0)
        {
            return;
        }

        auto diffuse_bsdf = diffuse_color / glm::pi<float>();
        auto diffuse_pdf  = NdotL / glm::pi<float>();


        auto H     = CuDiff::normalize(light_dir + view_dir);
        auto NdotH = CuDiff::max(CuDiff::dot(si.normal, H), 0.0f);
        auto VdotH = CuDiff::max(CuDiff::dot(H, view_dir), 0.0f);

        // Normal distribution
        auto NDF = D_GGX(NdotH, roughness);

        // Visibility
        auto V = V_SmithGGX(NdotL, NdotV, roughness);

        // Fresnel
        auto F = fresnel_schlick(metallic_color, VdotH);

        auto kS = F;
        auto kD = glm::vec3(1.0f) - kS;

        auto specular_bsdf = NDF * V * F;

        auto halfway_pdf             = NDF * NdotH;
        auto halfway_to_outgoing_pdf = atcg::warp_normal_to_reflected_direction_pdf(view_dir, H);    // 1 / (4*HdotV)
        auto specular_pdf            = halfway_pdf * halfway_to_outgoing_pdf;

        auto sample_probability = diffuse_probability * diffuse_pdf + specular_probability * specular_pdf;
        auto bsdf_weight        = (specular_bsdf + kD * diffuse_bsdf) * NdotL / (sample_probability + 1e-5f);

        glm::mat3 dbsdf_weightdalbedo =
            glm::mat3(bsdf_weight.derivative(0), bsdf_weight.derivative(1), bsdf_weight.derivative(2));
        glm::vec3 dbsdf_weightdm = bsdf_weight.derivative(3);
        glm::vec3 dbsdf_weightdr = bsdf_weight.derivative(4);


        glm::vec3 dLdalbedo = dLdbsdf * dbsdf_weightdalbedo + dLdwo * dwodalbedo;
        float dLdm          = glm::dot(dLdbsdf, dbsdf_weightdm) + glm::dot(dLdwo, dwodm);
        float dLdr          = glm::dot(dLdbsdf, dbsdf_weightdr) + glm::dot(dLdwo, dwodr);

        if(isnan(glm::length2(dLdalbedo)) || isnan(dLdm) || isnan(dLdr)) return;

        {
            DERIVATIVE_INTERPOLATION_VECTOR(dLdalbedo, sbt_data->diffuse_grad);
        }
        {
            DERIVATIVE_INTERPOLATION_SCALAR(dLdr, sbt_data->roughness_grad);
        }
        {
            DERIVATIVE_INTERPOLATION_SCALAR(dLdm, sbt_data->metallic_grad);
        }
    }

    // {
    //     const atcg::PBRBSDFData* sbt_data = *reinterpret_cast<const atcg::PBRBSDFData**>(optixGetSbtDataPointer());

    //     if(!sbt_data->optimizable) return;

    //     glm::vec3 albedo = sbt_data->diffuse_texture.read(si.uv);
    //     float m          = sbt_data->metallic_texture.read(si.uv);
    //     float r          = sbt_data->roughness_texture.read(si.uv);
    //     float R          = glm::max(r * r, 1e-3f);    // In the real time shaders, roughness is squared
    //     glm::vec3 D      = glm::lerp(albedo, glm::vec3(0), m) * si.color;

    //     glm::vec3 M = (1.0f - m) * glm::vec3(0.04f) + m * albedo;

    //     // ! SAMPLING
    //     glm::vec3 normal      = si.normal;
    //     glm::mat3 local_frame = atcg::Math::compute_local_frame(normal);

    //     float diffuse_sum          = glm::dot(D, glm::vec3(1));
    //     float metallic_sum         = glm::dot(M, glm::vec3(1));
    //     float denom                = (diffuse_sum + metallic_sum + 1e-5f);
    //     float diffuse_probability  = diffuse_sum / denom;
    //     float specular_probability = 1 - diffuse_probability;

    //     glm::vec3 ddiffuse_probabilitydD      = glm::vec3(metallic_sum / (denom * denom));
    //     glm::vec3 ddiffuse_probabilitydalbedo = (1.0f - m) * ddiffuse_probabilitydD;

    //     glm::vec3 ddiffuse_probabilitydM = glm::vec3(-diffuse_sum / (denom * denom));
    //     float ddiffuse_probabilitydm     = glm::dot(ddiffuse_probabilitydM, albedo - glm::vec3(0.04f));

    //     glm::vec3 dhdR = glm::vec3(0);

    //     glm::vec3 outgoing_dir = glm::vec3(0);
    //     if(rng.next1d() < diffuse_probability)
    //     {
    //         // Sample light direction from diffuse bsdf
    //         glm::vec3 local_outgoing_ray_dir = atcg::warp_square_to_hemisphere_cosine(rng.next2d());
    //         // Transform local outgoing direction from tangent space to world space
    //         outgoing_dir = local_frame * local_outgoing_ray_dir;

    //         // wo independent on r
    //     }
    //     else
    //     {
    //         // Sample light direction from specular bsdf
    //         auto uv                 = rng.next2d();
    //         glm::vec3 local_halfway = atcg::warp_square_to_hemisphere_ggx(uv, R);
    //         // Transform local halfway vector from tangent space to world space
    //         glm::vec3 halfway = local_frame * local_halfway;

    //         outgoing_dir = glm::reflect(si.incoming_direction, halfway);
    //         dhdR         = local_frame * warp_square_to_hemisphere_ggx_derivative(uv, R);
    //     }

    //     // ! EVAL
    //     glm::vec3 light_dir = outgoing_dir;
    //     glm::vec3 view_dir  = -si.incoming_direction;

    //     glm::vec3 H = glm::normalize(light_dir + view_dir);

    //     glm::mat3 dhdwo = (glm::mat3(1) - glm::outerProduct(H, H)) / glm::length(H);

    //     float NdotH = glm::max(glm::dot(si.normal, H), 0.0f);
    //     float NdotV = glm::max(glm::dot(si.normal, view_dir), 0.0f);
    //     float NdotL = glm::max(glm::dot(si.normal, light_dir), 0.0f);
    //     float VdotH = glm::max(glm::dot(H, view_dir), 0.0f);

    //     if(NdotL <= 0.0f || NdotV <= 0.0f) return;

    //     glm::mat3 dwodh = -glm::mat3(1) * 2.0f * VdotH - 2.0f * glm::outerProduct(H, si.incoming_direction);

    //     glm::vec3 dwodR = dwodh * dhdR;

    //     float NDF    = atcg::D_GGX(NdotH, R);
    //     float V      = atcg::V_SmithGGX(NdotL, NdotV, R);
    //     glm::vec3 F  = atcg::fresnel_schlick(M, VdotH);
    //     glm::vec3 kS = F;
    //     glm::vec3 kD = glm::vec3(1.0) - kS;

    //     glm::vec3 dFdM = detail::dfresnel_schlickdM(VdotH);
    //     float dNDFdR   = detail::dD_GGXdR(NdotH, R);
    //     float dVdR     = detail::dV_SmithGGXdR(NdotL, NdotV, R);

    //     glm::vec3 diffuse_bsdf  = glm::one_over_pi<float>() * D;
    //     glm::vec3 specular_bsdf = NDF * V * F;
    //     glm::vec3 bsdf_value    = (specular_bsdf + kD * diffuse_bsdf);

    //     float diffuse_pdf             = NdotL / glm::pi<float>();
    //     float halfway_pdf             = NDF * NdotH;
    //     float halfway_to_outgoing_pdf = atcg::warp_normal_to_reflected_direction_pdf(outgoing_dir, H);
    //     float specular_pdf            = halfway_pdf * halfway_to_outgoing_pdf;
    //     float p                       = diffuse_probability * diffuse_pdf + specular_probability * specular_pdf;

    //     glm::vec3 dbsdfdalbedo_v =
    //         kD * glm::one_over_pi<float>() * (1.0f - m) + m * dFdM * (V * NDF - diffuse_bsdf);    // Diagonal matrix
    //     glm::mat3 dbsdfdalbedo   = glm::mat3(glm::vec3(dbsdfdalbedo_v.x, 0, 0),
    //                                        glm::vec3(0, dbsdfdalbedo_v.y, 0),
    //                                        glm::vec3(0, 0, dbsdfdalbedo_v.z));
    //     glm::vec3 dpdalbedo      = ddiffuse_probabilitydalbedo * (diffuse_pdf - specular_pdf);
    //     glm::mat3 dweightdalbedo = (dbsdfdalbedo + glm::outerProduct(bsdf_value, dpdalbedo) / p) / p;
    //     glm::vec3 dLdalbedo      = dweightdalbedo * dLdbsdf;

    //     glm::vec3 dbsdfdr                = 2.0f * r * (F * dNDFdR * V + F * NDF * dVdR);
    //     float ddiffuse_pdfdr             = 2.0f * r * glm::one_over_pi<float>() * glm::dot(normal, dwodR);
    //     float dhalfway_pdfdR             = NDF * glm::dot(normal, dhdwo * dwodR) + NdotH * dNDFdR;
    //     float dhalfway_to_outgoing_pdfdR = -(glm::dot(H, dwodR) + glm::dot(light_dir, dhdR)) / (4.0f * VdotH *
    //     VdotH); float dspecular_pdfdr =
    //         2.0f * r * (halfway_to_outgoing_pdf * dhalfway_pdfdR + halfway_pdf * dhalfway_to_outgoing_pdfdR);
    //     float dpdr          = diffuse_probability * ddiffuse_pdfdr + specular_probability * dspecular_pdfdr;
    //     glm::vec3 dweightdr = (dbsdfdr + bsdf_value * dpdr / p) / p;
    //     float dLdr          = glm::dot(dweightdr, dLdbsdf);

    //     glm::vec3 dbsdfdm =
    //         (albedo - glm::vec3(0.04f)) * dFdM * (V * NDF - diffuse_bsdf) - albedo * kD * glm::one_over_pi<float>();
    //     float dpdm          = ddiffuse_probabilitydm * (diffuse_pdf - specular_pdf);
    //     glm::vec3 dweightdm = (dbsdfdm + bsdf_value * dpdm / p) / p;
    //     float dLdm          = glm::dot(dweightdm, dLdbsdf);

    //     {
    //         DERIVATIVE_INTERPOLATION_VECTOR(dLdalbedo, sbt_data->diffuse_grad);
    //     }
    //     {
    //         DERIVATIVE_INTERPOLATION_SCALAR(dLdr, sbt_data->roughness_grad);
    //     }
    //     {
    //         DERIVATIVE_INTERPOLATION_SCALAR(dLdm, sbt_data->metallic_grad);
    //     }
    // }

    // const atcg::PBRBSDFData* sbt_data = *reinterpret_cast<const atcg::PBRBSDFData**>(optixGetSbtDataPointer());

    // if(!sbt_data->optimizable) return;

    // glm::vec3 alpha         = sbt_data->diffuse_texture.read(si.uv);
    // float m                 = sbt_data->metallic_texture.read(si.uv);
    // float r                 = sbt_data->roughness_texture.read(si.uv);
    // float roughness         = glm::max(r * r, 1e-3f);    // In the real time shaders, roughness is squared
    // glm::vec3 diffuse_color = glm::lerp(alpha, glm::vec3(0), m) * si.color;

    // glm::vec3 metallic_color = (1.0f - m) * glm::vec3(0.04f) + m * alpha;

    // // Direction towards viewer
    // glm::vec3 view_dir = -si.incoming_direction;
    // glm::vec3 normal   = si.normal;

    // // Don't trace a new ray if surface is viewed from below
    // float NdotV = glm::dot(normal, view_dir);
    // if(NdotV <= 0)
    // {
    //     return;
    // }

    // // The matrix local_frame transforms a vector from the coordinate system where geom.N corresponds to the
    // z-axis to
    // // the world coordinate system.
    // glm::mat3 local_frame = atcg::Math::compute_local_frame(normal);

    // float diffuse_sum          = glm::dot(diffuse_color, glm::vec3(1));
    // float metallic_sum         = glm::dot(metallic_color, glm::vec3(1));
    // float denom                = (diffuse_sum + metallic_sum + 1e-5f);
    // float diffuse_probability  = diffuse_sum / denom;
    // float specular_probability = 1 - diffuse_probability;

    // glm::vec3 dpddiffuse  = glm::vec3(metallic_sum / (denom * denom));
    // glm::vec3 dpdmetallic = glm::vec3(-diffuse_sum / (denom * denom));

    // glm::vec3 dwodroughness = glm::vec3(0);
    // glm::mat3 dwoddiffuse   = glm::mat3(1);
    // glm::mat3 dwodmetallic  = glm::mat3(1);

    // if(rng.next1d() < diffuse_probability)
    // {
    //     // Sample light direction from diffuse bsdf
    //     glm::vec3 local_outgoing_ray_dir = atcg::warp_square_to_hemisphere_cosine(rng.next2d());
    //     // Transform local outgoing direction from tangent space to world space
    //     auto out_dir = local_frame * local_outgoing_ray_dir;

    //     dwoddiffuse  = glm::outerProduct(dpddiffuse, out_dir / diffuse_probability);
    //     dwodmetallic = glm::outerProduct(dpdmetallic, out_dir / diffuse_probability);
    // }
    // else
    // {
    //     // Sample light direction from specular bsdf
    //     auto uv                 = rng.next2d();
    //     glm::vec3 local_halfway = atcg::warp_square_to_hemisphere_ggx(uv, roughness);
    //     // Transform local halfway vector from tangent space to world space
    //     glm::vec3 halfway = local_frame * local_halfway;

    //     auto out_dir = glm::reflect(si.incoming_direction, halfway);
    //     float NdotL  = glm::dot(normal, out_dir);
    //     if(NdotL <= 0)
    //     {
    //         return;
    //     }

    //     float VdotH     = glm::dot(si.incoming_direction, halfway);
    //     glm::mat3 dwodh = -glm::mat3(1) * 2.0f * VdotH - 2.0f * glm::outerProduct(halfway,
    //     si.incoming_direction); glm::mat3 dhdh_ = local_frame; glm::vec3 dh_droughness =
    //     warp_square_to_hemisphere_ggx_derivative(uv, roughness);

    //     dwodroughness = dwodh * dhdh_ * dh_droughness;

    //     dwoddiffuse  = glm::outerProduct(-dpddiffuse, out_dir / specular_probability);
    //     dwodmetallic = glm::outerProduct(-dpdmetallic, out_dir / specular_probability);
    // }

    // // ? glm::mat3 dwidwo = glm::mat3(1) - 2.0f * glm::outerProduct(halfway, halfway);

    // glm::vec3 dwodr    = 2.0f * r * dwodroughness;
    // float dLdr         = glm::dot(out_grad, dwodr);
    // glm::vec3 dLdalpha = out_grad * dwoddiffuse * (1.0f - m);
    // float dLdm         = glm::dot(out_grad, dwodmetallic * (alpha - glm::vec3(0.04)));

    // {
    //     DERIVATIVE_INTERPOLATION_VECTOR(dLdalpha, sbt_data->diffuse_grad);
    // }
    // {
    //     DERIVATIVE_INTERPOLATION_SCALAR(dLdr, sbt_data->roughness_grad);
    // }
    // {
    //     DERIVATIVE_INTERPOLATION_SCALAR(dLdm, sbt_data->metallic_grad);
    // }
}
