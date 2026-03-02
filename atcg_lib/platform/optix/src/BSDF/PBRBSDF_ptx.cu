#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>
#include <Core/GlobalAtomicAdd.h>

#include <Math/Random.h>

#include <Math/Functions.h>
#include <Core/SurfaceInteraction.h>
#include <BSDF/BSDFVPtrTable.cuh>
#include <BSDF/PBRBSDFData.cuh>

#include <CuDiff/CuDiff.h>
#include <CuDiff/ext/glm.h>

namespace detail
{

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
                                                                      const atcg::SampledSpectrum& diffuse_color,
                                                                      const atcg::SampledSpectrum& specular_F0,
                                                                      const float& metallic,
                                                                      const float& roughness,
                                                                      const float& fixed_roughness,
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

    float diffuse_probability  = diffuse_color.sum() / (diffuse_color.sum() + specular_F0.sum() + 1e-5f);
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
        glm::vec3 local_halfway = atcg::warp_square_to_hemisphere_ggx(rng.next2d(), fixed_roughness);
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

    atcg::SampledSpectrum diffuse_bsdf = diffuse_color / glm::pi<float>();
    float diffuse_pdf                  = NdotL / glm::pi<float>();

    atcg::SampledSpectrum specular_bsdf = atcg::SampledSpectrum(0);
    float specular_pdf                  = 0;
    // Only compute specular component if specular_f0 is not zero!
    atcg::SampledSpectrum kD(1.0f);
    if(specular_F0.sum() > 1e-5f)
    {
        glm::vec3 halfway = glm::normalize(result.out_dir + view_dir);
        float HdotV       = glm::dot(halfway, result.out_dir);
        float NdotH       = glm::dot(halfway, normal);

        // Normal distribution
        float NDF = atcg::D_GGX(NdotH, roughness);

        // Visibility
        float V = atcg::V_SmithGGX(NdotL, NdotV, roughness);

        // Fresnel
        atcg::SampledSpectrum F = atcg::fresnel_schlick(specular_F0, HdotV);

        kD = (atcg::SampledSpectrum(1.0f) - F);

        specular_bsdf = NDF * V * F;

        float halfway_pdf = atcg::D_GGX(NdotH, fixed_roughness) * NdotH;
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
                                                                const atcg::SampledSpectrum& diffuse_color,
                                                                const atcg::SampledSpectrum& metallic_color,
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

    float NDF               = atcg::D_GGX(NdotH, roughness);
    float V                 = atcg::V_SmithGGX(NdotL, NdotV, roughness);
    atcg::SampledSpectrum F = atcg::fresnel_schlick(metallic_color, glm::max(glm::dot(H, view_dir), 0.0f));

    atcg::SampledSpectrum specular = NDF * V * F;

    atcg::SampledSpectrum kS = F;
    atcg::SampledSpectrum kD = atcg::SampledSpectrum(1.0) - kS;

    float diffuse_probability     = diffuse_color.sum() / (diffuse_color.sum() + metallic_color.sum() + 1e-5f);
    float specular_probability    = 1 - diffuse_probability;
    float diffuse_pdf             = NdotL / glm::pi<float>();
    float halfway_pdf             = NDF * NdotH;
    float halfway_to_outgoing_pdf = atcg::warp_normal_to_reflected_direction_pdf(outgoing_dir, H);    // 1 / (4*HdotV)
    float specular_pdf            = halfway_pdf * halfway_to_outgoing_pdf;

    result.bsdf_value         = (specular + kD * diffuse_color / glm::pi<float>()) * NdotL;
    result.sample_probability = diffuse_probability * diffuse_pdf + specular_probability * specular_pdf;
    result.flags =
        result.flags | (roughness < 0.1f ? atcg::BSDFComponentType::IdealReflection : atcg::BSDFComponentType::Any);

    return result;
}
}    // namespace detail

extern "C" __device__ atcg::BSDFSamplingResult
__direct_callable__sample_pbrbsdf(const atcg::SurfaceInteraction& si,
                                  const atcg::SampledWavelengths& wavelengths,
                                  atcg::PCG32& rng)
{
    const atcg::PBRBSDFData* sbt_data = *reinterpret_cast<const atcg::PBRBSDFData**>(optixGetSbtDataPointer());

    atcg::SampledSpectrum diffuse_color =
        atcg::SampledSpectrum::fromRGB(sbt_data->diffuse_texture.read(si.uv), wavelengths);
    float metallic        = sbt_data->metallic_texture.read(si.uv);
    float roughness       = sbt_data->roughness_texture.read(si.uv);
    roughness             = glm::max(roughness * roughness, 1e-3f);    // In the real time shaders, roughness is squared
    float fixed_roughness = sbt_data->fixed_roughness_texture.read(si.uv);
    fixed_roughness       = glm::max(fixed_roughness * fixed_roughness, 1e-3f);

    atcg::SampledSpectrum metallic_color = (1.0f - metallic) * atcg::SampledSpectrum(0.04f) + metallic * diffuse_color;
    diffuse_color = (1.0f - metallic) * diffuse_color * atcg::SampledSpectrum::fromRGB(si.color, wavelengths);

    return detail::samplePBR(si, diffuse_color, metallic_color, metallic, roughness, roughness, rng);
}

extern "C" __device__ atcg::BSDFEvalResult __direct_callable__eval_pbrbsdf(const atcg::SurfaceInteraction& si,
                                                                           const glm::vec3& outgoing_dir,
                                                                           const atcg::SampledWavelengths& wavelengths)
{
    const atcg::PBRBSDFData* sbt_data = *reinterpret_cast<const atcg::PBRBSDFData**>(optixGetSbtDataPointer());

    atcg::SampledSpectrum diffuse_color =
        atcg::SampledSpectrum::fromRGB(sbt_data->diffuse_texture.read(si.uv), wavelengths);
    float metallic  = sbt_data->metallic_texture.read(si.uv);
    float roughness = sbt_data->roughness_texture.read(si.uv);
    roughness       = glm::max(roughness * roughness, 1e-3f);    // In the real time shaders, roughness is squared

    atcg::SampledSpectrum metallic_color = (1.0f - metallic) * atcg::SampledSpectrum(0.04f) + metallic * diffuse_color;
    diffuse_color = (1.0f - metallic) * diffuse_color * atcg::SampledSpectrum::fromRGB(si.color, wavelengths);


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
    auto cos_theta = CuDiff::sqrt(CuDiff::max(1e-12f, (1.0f - uv.x) / (1.0f + (roughness * roughness - 1.0f) * uv.x)));
    auto sin_theta = CuDiff::sqrt(CuDiff::max(1e-12f, 1.0f - cos_theta * cos_theta));
    float phi      = 2.0f * glm::pi<float>() * uv.y;

    auto x = sin_theta * glm::cos(phi);
    auto y = sin_theta * glm::sin(phi);
    auto z = cos_theta;

    auto res = CuDiff::wrap(x, y, z);

    return res;
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
__direct_callable__sample_forward_pbrbsdf(const atcg::DualSurfaceInteraction& si,
                                          const atcg::SampledWavelengths& wavelengths,
                                          atcg::PCG32& rng)
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
        result.out_dir = apply_local_frame(local_frame, CuDiff::Dual<6, glm::vec3>(local_outgoing_ray_dir));
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
        result.sample_probability = CuDiff::Dual<6, float>(0.0f);
        return result;
    }

    auto diffuse_bsdf = diffuse_color / glm::pi<float>();
    auto diffuse_pdf  = NdotL / glm::pi<float>();

    CuDiff::Dual<6, glm::vec3> specular_bsdf = CuDiff::Dual<6, glm::vec3>(glm::vec3(0.0f));
    CuDiff::Dual<6, float> specular_pdf      = CuDiff::Dual<6, float>(0.0f);
    // Only compute specular component if specular_f0 is not zero!
    CuDiff::Dual<6, glm::vec3> kD = CuDiff::Dual<6, glm::vec3>(glm::vec3(1.0f));
    if(CuDiff::dot(metallic_color, metallic_color) > 1e-6f)
    {
        auto halfway = CuDiff::normalize(result.out_dir + view_dir);
        auto HdotV   = CuDiff::dot(halfway, result.out_dir);
        auto NdotH   = CuDiff::dot(halfway, normal);

        // Normal distribution
        auto NDF = atcg::D_GGX(NdotH, roughness);

        // Visibility
        auto V = atcg::V_SmithGGX(NdotL, NdotV, roughness);

        // Fresnel
        auto F = atcg::fresnel_schlick(metallic_color, HdotV);

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
__direct_callable__eval_forward_pbrbsdf(const atcg::DualSurfaceInteraction& si,
                                        const CuDiff::Dual<6, glm::vec3>& outgoing_dir,
                                        const atcg::SampledWavelengths& wavelengths)
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


    atcg::BSDFDualEvalResult result;

    auto light_dir = outgoing_dir;
    auto view_dir  = -si.incoming_direction;

    auto H = CuDiff::normalize(light_dir + view_dir);

    auto NdotH = CuDiff::max(CuDiff::dot(si.normal, H), 0.0f);
    auto NdotV = CuDiff::max(CuDiff::dot(si.normal, view_dir), 0.0f);
    auto NdotL = CuDiff::max(CuDiff::dot(si.normal, light_dir), 0.0f);

    if(NdotL <= 0.0f || NdotV <= 0.0f) return result;

    auto NDF = atcg::D_GGX(NdotH, roughness);
    auto V   = atcg::V_SmithGGX(NdotL, NdotV, roughness);
    auto F   = atcg::fresnel_schlick(metallic_color, CuDiff::max(CuDiff::dot(H, view_dir), 0.0f));

    auto specular = NDF * V * F;

    auto kS = F;
    auto kD = glm::vec3(1.0f) - kS;

    float diffuse_probability =
        glm::dot(diffuse_color.val(), glm::vec3(1.0f)) /
        (glm::dot(diffuse_color.val(), glm::vec3(1.0f)) + glm::dot(metallic_color.val(), glm::vec3(1.0f)) + 1e-5f);
    float specular_probability    = 1 - diffuse_probability;
    float diffuse_pdf             = NdotL / glm::pi<float>();
    float halfway_pdf             = NDF * NdotH;
    float halfway_to_outgoing_pdf = atcg::warp_normal_to_reflected_direction_pdf(outgoing_dir, H);    // 1 / (4*HdotV)
    float specular_pdf            = halfway_pdf * halfway_to_outgoing_pdf;

    result.bsdf_value         = (specular + kD * diffuse_color / glm::pi<float>()) * NdotL;
    result.sample_probability = diffuse_probability * diffuse_pdf + specular_probability * specular_pdf;
    result.flags =
        result.flags | (roughness < 0.1f ? atcg::BSDFComponentType::IdealReflection : atcg::BSDFComponentType::Any);

    return result;
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
    if(x0 == x1 && y0 == y1)                                                                                           \
    {                                                                                                                  \
        float* inp_grad##_p = (float*)texture.getTexelPtr(glm::ivec2(x0, y0));                                         \
        atcg::globalAtomicAdd(inp_grad##_p + 0, inp_grad.x);                                                           \
        atcg::globalAtomicAdd(inp_grad##_p + 1, inp_grad.y);                                                           \
        atcg::globalAtomicAdd(inp_grad##_p + 2, inp_grad.z);                                                           \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        float tx                 = fx - x0;                                                                            \
        float ty                 = fy - y0;                                                                            \
        glm::vec3 inp_grad##_c00 = inp_grad * (1.0f - ty) * (1.0f - tx);                                               \
        glm::vec3 inp_grad##_c10 = inp_grad * (1.0f - ty) * (tx);                                                      \
        glm::vec3 inp_grad##_c01 = inp_grad * (ty) * (1.0f - tx);                                                      \
        glm::vec3 inp_grad##_c11 = inp_grad * (ty) * (tx);                                                             \
        float* inp_grad##_c00_p  = (float*)texture.getTexelPtr(glm::ivec2(x0, y0));                                    \
        float* inp_grad##_c10_p  = (float*)texture.getTexelPtr(glm::ivec2(x1, y0));                                    \
        float* inp_grad##_c01_p  = (float*)texture.getTexelPtr(glm::ivec2(x0, y1));                                    \
        float* inp_grad##_c11_p  = (float*)texture.getTexelPtr(glm::ivec2(x1, y1));                                    \
        atcg::globalAtomicAdd(inp_grad##_c00_p + 0, inp_grad##_c00.x);                                                 \
        atcg::globalAtomicAdd(inp_grad##_c00_p + 1, inp_grad##_c00.y);                                                 \
        atcg::globalAtomicAdd(inp_grad##_c00_p + 2, inp_grad##_c00.z);                                                 \
        atcg::globalAtomicAdd(inp_grad##_c10_p + 0, inp_grad##_c10.x);                                                 \
        atcg::globalAtomicAdd(inp_grad##_c10_p + 1, inp_grad##_c10.y);                                                 \
        atcg::globalAtomicAdd(inp_grad##_c10_p + 2, inp_grad##_c10.z);                                                 \
        atcg::globalAtomicAdd(inp_grad##_c01_p + 0, inp_grad##_c01.x);                                                 \
        atcg::globalAtomicAdd(inp_grad##_c01_p + 1, inp_grad##_c01.y);                                                 \
        atcg::globalAtomicAdd(inp_grad##_c01_p + 2, inp_grad##_c01.z);                                                 \
        atcg::globalAtomicAdd(inp_grad##_c11_p + 0, inp_grad##_c11.x);                                                 \
        atcg::globalAtomicAdd(inp_grad##_c11_p + 1, inp_grad##_c11.y);                                                 \
        atcg::globalAtomicAdd(inp_grad##_c11_p + 2, inp_grad##_c11.z);                                                 \
    }                                                                                                                  \
    void(0)

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
    if(x0 == x1 && y0 == y1)                                                                                           \
    {                                                                                                                  \
        float* inp_grad##_p = (float*)texture.getTexelPtr(glm::ivec2(x0, y0));                                         \
        atcg::globalAtomicAdd(inp_grad##_p, inp_grad);                                                                 \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        float tx                = fx - x0;                                                                             \
        float ty                = fy - y0;                                                                             \
        float inp_grad##_c00    = inp_grad * (1.0f - ty) * (1.0f - tx);                                                \
        float inp_grad##_c10    = inp_grad * (1.0f - ty) * (tx);                                                       \
        float inp_grad##_c01    = inp_grad * (ty) * (1.0f - tx);                                                       \
        float inp_grad##_c11    = inp_grad * (ty) * (tx);                                                              \
        float* inp_grad##_c00_p = (float*)texture.getTexelPtr(glm::ivec2(x0, y0));                                     \
        float* inp_grad##_c10_p = (float*)texture.getTexelPtr(glm::ivec2(x1, y0));                                     \
        float* inp_grad##_c01_p = (float*)texture.getTexelPtr(glm::ivec2(x0, y1));                                     \
        float* inp_grad##_c11_p = (float*)texture.getTexelPtr(glm::ivec2(x1, y1));                                     \
        atcg::globalAtomicAdd(inp_grad##_c00_p + 0, inp_grad##_c00);                                                   \
        atcg::globalAtomicAdd(inp_grad##_c10_p + 0, inp_grad##_c10);                                                   \
        atcg::globalAtomicAdd(inp_grad##_c01_p + 0, inp_grad##_c01);                                                   \
        atcg::globalAtomicAdd(inp_grad##_c11_p + 0, inp_grad##_c11);                                                   \
    }                                                                                                                  \
    void(0)

extern "C" __device__ void __direct_callable__eval_backward_pbrbsdf(const atcg::SurfaceInteraction& si,
                                                                    const glm::vec3& outgoing_dir,
                                                                    const glm::vec3& dLdbsdf)
{
    {
        const atcg::PBRBSDFData* sbt_data = *reinterpret_cast<const atcg::PBRBSDFData**>(optixGetSbtDataPointer());

        if(!sbt_data->optimize_diffuse && !sbt_data->optimize_metallic && !sbt_data->optimize_roughness) return;

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

        auto NDF = atcg::D_GGX(NdotH, roughness);
        auto V   = atcg::V_SmithGGX(NdotL, NdotV, roughness);
        auto F   = atcg::fresnel_schlick(metallic_color, VdotH);
        auto kS  = F;
        auto kD  = glm::vec3(1.0) - kS;

        auto specular = NDF * V * F;

        auto bsdf_value = (specular + kD * diffuse_color / glm::pi<float>()) * NdotL;

        if(sbt_data->optimize_diffuse)
        {
            glm::mat3 dbsdfdalpha =
                glm::mat3(bsdf_value.derivative(0), bsdf_value.derivative(1), bsdf_value.derivative(2));
            glm::vec3 dLdalbedo = dLdbsdf * dbsdfdalpha;
            DERIVATIVE_INTERPOLATION_VECTOR(dLdalbedo, sbt_data->diffuse_grad);
        }

        if(sbt_data->optimize_roughness)
        {
            glm::vec3 dbsdfdr = bsdf_value.derivative(4);
            float dLdr        = glm::dot(dbsdfdr, dLdbsdf);
            DERIVATIVE_INTERPOLATION_SCALAR(dLdr, sbt_data->roughness_grad);
        }

        if(sbt_data->optimize_metallic)
        {
            glm::vec3 dbsdfdm = bsdf_value.derivative(3);
            float dLdm        = glm::dot(dbsdfdm, dLdbsdf);
            DERIVATIVE_INTERPOLATION_SCALAR(dLdm, sbt_data->metallic_grad);
        }
    }
}

extern "C" __device__ void __direct_callable__sample_backward_pbrbsdf(const atcg::SurfaceInteraction& si,
                                                                      atcg::PCG32& rng,
                                                                      const glm::vec3& dLdbsdf,
                                                                      const glm::vec3& dLdwo_)
{
    {
        const atcg::PBRBSDFData* sbt_data = *reinterpret_cast<const atcg::PBRBSDFData**>(optixGetSbtDataPointer());

        if(!sbt_data->optimize_diffuse && !sbt_data->optimize_roughness && !sbt_data->optimize_metallic) return;

        glm::vec3 dLdwo = dLdwo_;
        if(!isfinite(dLdwo.x) || !isfinite(dLdwo.y) || !isfinite(dLdwo.z))
        {
            dLdwo = glm::vec3(0.0f);
        }

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
            out_dir = CuDiff::Dual<5, glm::vec3>(local_frame * local_outgoing_ray_dir);
        }
        else
        {
            // Sample light direction from specular bsdf
            auto local_halfway = warp_square_to_hemisphere_ggx(rng.next2d(), roughness);
            // Transform local halfway vector from tangent space to world space
            auto halfway = local_frame * local_halfway;
            out_dir      = CuDiff::reflect(si.incoming_direction, halfway);
        }

        glm::mat3 dwodalbedo = glm::mat3(out_dir.derivative(0), out_dir.derivative(1), out_dir.derivative(2));
        glm::vec3 dwodm      = out_dir.derivative(3);
        glm::vec3 dwodr      = out_dir.derivative(4);

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
        auto NDF = atcg::D_GGX(NdotH, roughness);

        // Visibility
        auto V = atcg::V_SmithGGX(NdotL, NdotV, roughness);

        // Fresnel
        auto F = atcg::fresnel_schlick(metallic_color, VdotH);

        auto kS = F;
        auto kD = glm::vec3(1.0f) - kS;

        auto specular_bsdf = NDF * V * F;

        auto halfway_pdf             = NDF * NdotH;
        auto halfway_to_outgoing_pdf = atcg::warp_normal_to_reflected_direction_pdf(view_dir, H);    // 1 / (4*HdotV)
        auto specular_pdf            = halfway_pdf * halfway_to_outgoing_pdf;

        auto sample_probability = diffuse_probability * diffuse_pdf + specular_probability * specular_pdf;
        auto bsdf_value         = (specular_bsdf + kD * diffuse_bsdf) * NdotL;
        auto bsdf_weight        = bsdf_value / (sample_probability + 1e-5f);


        // if(!isfinite(dLdalbedo.x) || !isfinite(dLdalbedo.y) || !isfinite(dLdalbedo.z))
        // {
        //     dLdalbedo = glm::vec3(0.0f);
        // }

        // if(!isfinite(dLdm))
        // {
        //     dLdm = 0.0f;
        // }

        // if(!isfinite(dLdr))
        // {
        //     dLdr = 0.0f;
        // }

        if(sbt_data->optimize_diffuse)
        {
            glm::mat3 dbsdf_weightdalbedo =
                glm::mat3(bsdf_weight.derivative(0), bsdf_weight.derivative(1), bsdf_weight.derivative(2));
            glm::vec3 partial_wo = dLdwo * dwodalbedo;
            glm::vec3 dLdalbedo  = dLdbsdf * dbsdf_weightdalbedo;

            // if(isfinite(partial_wo.x) && isfinite(partial_wo.y) && isfinite(partial_wo.z))
            // {
            dLdalbedo += partial_wo;
            // }

            if(!isfinite(dLdalbedo.x) || !isfinite(dLdalbedo.y) || !isfinite(dLdalbedo.z))
            {
                dLdalbedo = glm::vec3(0.0f);
            }

            DERIVATIVE_INTERPOLATION_VECTOR(dLdalbedo, sbt_data->diffuse_grad);
        }

        if(sbt_data->optimize_roughness)
        {
            glm::vec3 dbsdf_weightdr = bsdf_weight.derivative(4);
            float partial_r          = glm::dot(dLdwo, dwodr);
            float dLdr               = glm::dot(dLdbsdf, dbsdf_weightdr);

            float dbsdf     = dLdr;
            float dsampling = partial_r;

            // if(isfinite(partial_r))
            // {
            dLdr += partial_r;
            // }

            if(!isfinite(dLdr))
            {
                dLdr = 0.0f;
            }

            DERIVATIVE_INTERPOLATION_SCALAR(dLdr, sbt_data->roughness_grad);

            atomicAdd(sbt_data->roughness_bsdf, dbsdf);
            atomicAdd(sbt_data->roughness_sampling, dsampling);
        }

        if(sbt_data->optimize_metallic)
        {
            glm::vec3 dbsdf_weightdm = bsdf_weight.derivative(3);
            float partial_m          = glm::dot(dLdwo, dwodm);
            float dLdm               = glm::dot(dLdbsdf, dbsdf_weightdm);

            // if(isfinite(partial_m))
            // {
            dLdm += partial_m;
            // }

            if(!isfinite(dLdm))
            {
                dLdm = 0.0f;
            }

            DERIVATIVE_INTERPOLATION_SCALAR(dLdm, sbt_data->metallic_grad);
        }
    }
}
