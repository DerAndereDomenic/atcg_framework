#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>

#include <Utils/HostDevice.h>
#include <DataStructure/SurfaceInteraction.h>
#include <Material/BSDFVPtrTable.h>
#include <Material/OpaqueMaterialData.h>
#include <BSDF/BSDFFunctions.h>
#include <BSDF/Sampling.h>

#include <DataStructure/Frame.h>

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
                                                                      atcg::PCG32& rng)
{
    atcg::BSDFSamplingResult result;

    // Direction towards viewer
    glm::vec3 view_dir = -si.incoming_direction;
    glm::vec3 normal   = atcg::faceForward(si.normal, view_dir);

    // Don't trace a new ray if surface is viewed from below
    float NdotV = glm::dot(normal, view_dir);
    if(NdotV <= 0)
    {
        return result;
    }

    // The matrix local_frame transforms a vector from the coordinate system where geom.N corresponds to the z-axis to
    // the world coordinate system.
    atcg::Frame local_frame = atcg::Frame(normal);

    float diffuse_probability  = diffuse_color.sum() / (diffuse_color.sum() + specular_F0.sum() + 1e-5f);
    float specular_probability = 1 - diffuse_probability;

    if(rng.next1d() < diffuse_probability)
    {
        // Sample light direction from diffuse bsdf
        atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_COSINE> strategy;
        glm::vec3 local_outgoing_ray_dir = strategy.sample(rng.next2d());
        // Transform local outgoing direction from tangent space to world space
        result.out_dir = local_frame.toWorld(local_outgoing_ray_dir);
        result.flags   = atcg::MaterialFlag::DiffuseReflection;
    }
    else
    {
        // Sample light direction from specular bsdf
        atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_GGX> strategy(roughness);
        glm::vec3 local_halfway = strategy.sample(rng.next2d());
        // Transform local halfway vector from tangent space to world space
        glm::vec3 halfway = local_frame.toWorld(local_halfway);
        result.out_dir    = glm::reflect(si.incoming_direction, halfway);

        result.flags = (roughness < 0.01f ? atcg::MaterialFlag::IdealReflection : atcg::MaterialFlag::GlossyReflection);
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

        float halfway_pdf = NDF * NdotH;
        float halfway_to_outgoing_pdf =
            atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_GGX>::warp_halfway_to_reflected_direction_pdf(
                result.out_dir,
                halfway);    // 1 / (4*HdotV)
        specular_pdf = halfway_pdf * halfway_to_outgoing_pdf;
    }

    result.sample_probability = diffuse_probability * diffuse_pdf + specular_probability * specular_pdf;
    result.bsdf_weight        = (specular_bsdf + kD * diffuse_bsdf) * NdotL / (result.sample_probability + 1e-5f);

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
    glm::vec3 normal    = atcg::faceForward(si.normal, view_dir);

    glm::vec3 H = glm::normalize(light_dir + view_dir);

    float NdotH = glm::max(glm::dot(normal, H), 0.0f);
    float NdotV = glm::max(glm::dot(normal, view_dir), 0.0f);
    float NdotL = glm::max(glm::dot(normal, light_dir), 0.0f);

    if(NdotL <= 0.0f || NdotV <= 0.0f) return result;

    float NDF               = atcg::D_GGX(NdotH, roughness);
    float V                 = atcg::V_SmithGGX(NdotL, NdotV, roughness);
    atcg::SampledSpectrum F = atcg::fresnel_schlick(metallic_color, glm::max(glm::dot(H, view_dir), 0.0f));

    atcg::SampledSpectrum specular = NDF * V * F;

    atcg::SampledSpectrum kS = F;
    atcg::SampledSpectrum kD = atcg::SampledSpectrum(1.0) - kS;

    float diffuse_probability  = diffuse_color.sum() / (diffuse_color.sum() + metallic_color.sum() + 1e-5f);
    float specular_probability = 1 - diffuse_probability;
    float diffuse_pdf          = NdotL / glm::pi<float>();
    float halfway_pdf          = NDF * NdotH;
    float halfway_to_outgoing_pdf =
        atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_GGX>::warp_halfway_to_reflected_direction_pdf(
            outgoing_dir,
            H);    // 1 / (4*HdotV)
    float specular_pdf = halfway_pdf * halfway_to_outgoing_pdf;

    result.bsdf_value         = (specular + kD * diffuse_color / glm::pi<float>()) * NdotL;
    result.sample_probability = diffuse_probability * diffuse_pdf + specular_probability * specular_pdf;
    result.flags = (roughness < 0.01f ? atcg::MaterialFlag::IdealReflection
                                      : atcg::MaterialFlag::GlossyReflection | atcg::MaterialFlag::DiffuseReflection);

    return result;
}
}    // namespace detail

extern "C" __device__ atcg::BSDFSamplingResult
__direct_callable__sample_pbrbsdf(const atcg::SurfaceInteraction& si,
                                  const atcg::SampledWavelengths& wavelengths,
                                  atcg::PCG32& rng)
{
    const atcg::OpaqueMaterialData* sbt_data =
        *reinterpret_cast<const atcg::OpaqueMaterialData**>(optixGetSbtDataPointer());

    atcg::SampledSpectrum diffuse_color =
        atcg::SampledSpectrum::fromRGB(sbt_data->diffuse_texture.read(si.uv), wavelengths);
    float metallic  = sbt_data->metallic_texture.read(si.uv);
    float roughness = sbt_data->roughness_texture.read(si.uv);
    roughness       = glm::max(roughness * roughness, 1e-3f);    // In the real time shaders, roughness is squared

    atcg::SampledSpectrum metallic_color = (1.0f - metallic) * atcg::SampledSpectrum(0.04f) + metallic * diffuse_color;
    diffuse_color = (1.0f - metallic) * diffuse_color * atcg::SampledSpectrum::fromRGB(si.color, wavelengths);

    return detail::samplePBR(si, diffuse_color, metallic_color, metallic, roughness, rng);
}

extern "C" __device__ atcg::BSDFEvalResult __direct_callable__eval_pbrbsdf(const atcg::SurfaceInteraction& si,
                                                                           const glm::vec3& outgoing_dir,
                                                                           const atcg::SampledWavelengths& wavelengths)
{
    const atcg::OpaqueMaterialData* sbt_data =
        *reinterpret_cast<const atcg::OpaqueMaterialData**>(optixGetSbtDataPointer());

    atcg::SampledSpectrum diffuse_color =
        atcg::SampledSpectrum::fromRGB(sbt_data->diffuse_texture.read(si.uv), wavelengths);
    float metallic  = sbt_data->metallic_texture.read(si.uv);
    float roughness = sbt_data->roughness_texture.read(si.uv);
    roughness       = glm::max(roughness * roughness, 1e-3f);    // In the real time shaders, roughness is squared

    atcg::SampledSpectrum metallic_color = (1.0f - metallic) * atcg::SampledSpectrum(0.04f) + metallic * diffuse_color;
    diffuse_color = (1.0f - metallic) * diffuse_color * atcg::SampledSpectrum::fromRGB(si.color, wavelengths);


    return detail::evalPBR(si, outgoing_dir, diffuse_color, metallic_color, roughness, metallic);
}