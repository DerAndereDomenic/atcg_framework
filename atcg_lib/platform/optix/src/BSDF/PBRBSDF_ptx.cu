#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>

#include <Math/Functions.h>
#include <Core/SurfaceInteraction.h>
#include <BSDF/BSDFVPtrTable.cuh>
#include <BSDF/PBRBSDFData.cuh>

namespace detail
{

ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec3 dfresnel_schlick(const float VdotH)
{
    return glm::vec3(1.0f) - glm::pow(glm::max(0.0f, 1.0f - VdotH), 5.0f);
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE float dD_GGX(const float NdotH, const float roughness)
{
    float r      = roughness;
    float r2     = r * r;
    float NdotH2 = NdotH * NdotH;
    float num    = 2.0f * r * (-2.0f * NdotH2 * r2 + NdotH2 * (r2 - 1.0f) + 1.0f);
    float denom  = (NdotH2 * (r2 - 1.0f) + 1.0f);

    return num / (glm::pi<float>() * denom * denom * denom + 1e-5f);
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T dV_SmithGGX(T NdotL, T NdotV, T alpha, T eps = 1e-8f)
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

    return num / (denom + T(1e-5));
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

extern "C" __device__ void __direct_callable__grad_pbrbsdf(const atcg::SurfaceInteraction& si,
                                                           const glm::vec3& outgoing_dir,
                                                           const glm::vec3& out_grad)
{
    const atcg::PBRBSDFData* sbt_data = *reinterpret_cast<const atcg::PBRBSDFData**>(optixGetSbtDataPointer());

    if(!sbt_data->optimizable) return;

    glm::vec3 alpha         = sbt_data->diffuse_texture.read(si.uv);
    float m                 = sbt_data->metallic_texture.read(si.uv);
    float r                 = sbt_data->roughness_texture.read(si.uv);
    float roughness         = glm::max(r * r, 1e-3f);    // In the real time shaders, roughness is squared
    glm::vec3 diffuse_color = glm::lerp(alpha, glm::vec3(0), m) * si.color;

    glm::vec3 metallic_color = (1.0f - m) * glm::vec3(0.04f) + m * alpha;

    glm::vec3 light_dir = outgoing_dir;
    glm::vec3 view_dir  = -si.incoming_direction;

    glm::vec3 H = glm::normalize(light_dir + view_dir);

    float NdotH = glm::max(glm::dot(si.normal, H), 0.0f);
    float NdotV = glm::max(glm::dot(si.normal, view_dir), 0.0f);
    float NdotL = glm::max(glm::dot(si.normal, light_dir), 0.0f);
    float VdotH = glm::max(glm::dot(H, view_dir), 0.0f);

    if(NdotL <= 0.0f || NdotV <= 0.0f) return;

    float NDF    = atcg::D_GGX(NdotH, roughness);
    float V      = atcg::V_SmithGGX(NdotL, NdotV, roughness);
    glm::vec3 F  = atcg::fresnel_schlick(metallic_color, VdotH);
    glm::vec3 kS = F;
    glm::vec3 kD = glm::vec3(1.0) - kS;

    glm::vec3 dFdM = detail::dfresnel_schlick(VdotH);
    float dNDFdR   = detail::dD_GGX(NdotH, roughness);
    float dVdR     = detail::dV_SmithGGX(NdotL, NdotV, roughness);

    glm::vec3 diffuse_bsdf = glm::one_over_pi<float>() * diffuse_color;

    glm::vec3 grad_albedo =
        (kD * glm::one_over_pi<float>() * (1.0f - m) + m * dFdM * (V * NDF - diffuse_bsdf)) * out_grad;

    float grad_roughness = glm::dot((2.0f * r * (F * dNDFdR * V + F * NDF * dVdR)), out_grad);
    float grad_metallic  = glm::dot(
        ((alpha - glm::vec3(0.04f)) * dFdM * (V * NDF - diffuse_bsdf) - alpha * kD * glm::one_over_pi<float>()),
        out_grad);

    // glm::vec2 uv = sbt_data->diffuse_grad.clamp_uv(si.uv);
    // float fx     = uv.x * (sbt_data->diffuse_grad.getSpecification().width - 1);
    // float fy     = uv.y * (sbt_data->diffuse_grad.getSpecification().height - 1);

    // int x0 = static_cast<int>(glm::floor(fx));
    // int y0 = static_cast<int>(glm::floor(fy));
    // int x1 = glm::min(x0 + 1, (int)sbt_data->diffuse_grad.getSpecification().width - 1);    // TODO: wrap
    // int y1 = glm::min(y0 + 1, (int)sbt_data->diffuse_grad.getSpecification().height - 1);

    // float tx = fx - x0;
    // float ty = fy - y0;

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

    {
        DERIVATIVE_INTERPOLATION_VECTOR(grad_albedo, sbt_data->diffuse_grad);
    }
    {
        DERIVATIVE_INTERPOLATION_SCALAR(grad_roughness, sbt_data->roughness_grad);
    }
    {
        DERIVATIVE_INTERPOLATION_SCALAR(grad_metallic, sbt_data->metallic_grad);
    }

#undef DERIVATIVE_INTERPOLATION

    // glm::vec3 grad = glm::one_over_pi<float>() * out_grad;

    // glm::vec2 uv = sbt_data->diffuse_grad.clamp_uv(si.uv);
    // float fx     = uv.x * (sbt_data->diffuse_grad.getSpecification().width - 1);
    // float fy     = uv.y * (sbt_data->diffuse_grad.getSpecification().height - 1);

    // int x0 = static_cast<int>(glm::floor(fx));
    // int y0 = static_cast<int>(glm::floor(fy));
    // int x1 = glm::min(x0 + 1, (int)sbt_data->diffuse_grad.getSpecification().width - 1);    // TODO: wrap
    // int y1 = glm::min(y0 + 1, (int)sbt_data->diffuse_grad.getSpecification().height - 1);

    // float tx = fx - x0;
    // float ty = fy - y0;

    // glm::vec3 grad_c00 = grad * (1.0f - ty) * (1.0f - tx);
    // glm::vec3 grad_c10 = grad * (1.0f - ty) * (tx);
    // glm::vec3 grad_c01 = grad * (ty) * (1.0f - tx);
    // glm::vec3 grad_c11 = grad * (ty) * (tx);

    // float* c00_p = (float*)sbt_data->diffuse_grad.getTexelPtr(glm::ivec2(x0, y0));
    // float* c10_p = (float*)sbt_data->diffuse_grad.getTexelPtr(glm::ivec2(x1, y0));
    // float* c01_p = (float*)sbt_data->diffuse_grad.getTexelPtr(glm::ivec2(x0, y1));
    // float* c11_p = (float*)sbt_data->diffuse_grad.getTexelPtr(glm::ivec2(x1, y1));

    // atomicAdd(c00_p + 0, grad_c00.x);
    // atomicAdd(c00_p + 1, grad_c00.y);
    // atomicAdd(c00_p + 2, grad_c00.z);

    // atomicAdd(c10_p + 0, grad_c10.x);
    // atomicAdd(c10_p + 1, grad_c10.y);
    // atomicAdd(c10_p + 2, grad_c10.z);

    // atomicAdd(c01_p + 0, grad_c01.x);
    // atomicAdd(c01_p + 1, grad_c01.y);
    // atomicAdd(c01_p + 2, grad_c01.z);

    // atomicAdd(c11_p + 0, grad_c11.x);
    // atomicAdd(c11_p + 1, grad_c11.y);
    // atomicAdd(c11_p + 2, grad_c11.z);
}