#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>

#include <Pathtracing/SurfaceInteraction.h>
#include <Pathtracing/BSDF/BSDFModels.cuh>

extern "C" __device__ atcg::BSDFSamplingResult __direct_callable__sample_pbrbsdf(const atcg::SurfaceInteraction& si,
                                                                                 atcg::PCG32& rng)
{
    const atcg::PBRBSDFData* sbt_data = *reinterpret_cast<const atcg::PBRBSDFData**>(optixGetSbtDataPointer());

    float4 color_u          = tex2D<float4>(sbt_data->diffuse_texture, si.uv.x, si.uv.y);
    glm::vec3 diffuse_color = glm::vec3(color_u.x, color_u.y, color_u.z);
    float metallic          = tex2D<float>(sbt_data->metallic_texture, si.uv.x, si.uv.y);
    float roughness         = tex2D<float>(sbt_data->roughness_texture, si.uv.x, si.uv.y);
    roughness = glm::max(roughness * roughness, 1e-3f);    // In the real time shaders, roughness is squared

    glm::vec3 metallic_color = (1.0f - metallic) * glm::vec3(0.04f) + metallic * diffuse_color;

    atcg::BSDFSamplingResult result = atcg::sampleGGX(si, diffuse_color, metallic_color, metallic, roughness, rng);

    return result;
}

extern "C" __device__ atcg::BSDFEvalResult __direct_callable__eval_pbrbsdf(const atcg::SurfaceInteraction& si,
                                                                           const glm::vec3& outgoing_dir)
{
    const atcg::PBRBSDFData* sbt_data = *reinterpret_cast<const atcg::PBRBSDFData**>(optixGetSbtDataPointer());

    float4 color_u          = tex2D<float4>(sbt_data->diffuse_texture, si.uv.x, si.uv.y);
    glm::vec3 diffuse_color = glm::vec3(color_u.x, color_u.y, color_u.z);
    float metallic          = tex2D<float>(sbt_data->metallic_texture, si.uv.x, si.uv.y);
    float roughness         = tex2D<float>(sbt_data->roughness_texture, si.uv.x, si.uv.y);
    roughness = glm::max(roughness * roughness, 1e-3f);    // In the real time shaders, roughness is squared

    glm::vec3 metallic_color = (1.0f - metallic) * glm::vec3(0.04f) + metallic * diffuse_color;

    glm::vec3 light_dir = outgoing_dir;
    glm::vec3 view_dir  = -si.incoming_direction;

    glm::vec3 H = glm::normalize(light_dir + view_dir);

    float NdotH = glm::max(glm::dot(si.normal, H), 0.0f);
    float NdotV = glm::max(glm::dot(si.normal, view_dir), 0.0f);
    float NdotL = glm::max(glm::dot(si.normal, light_dir), 0.0f);

    float NDF   = atcg::D_GGX(NdotH, roughness);
    float G     = atcg::geometrySmith(NdotL, NdotV, roughness);
    glm::vec3 F = atcg::fresnel_schlick(metallic_color, max(dot(H, view_dir), 0.0));

    glm::vec3 numerator = NDF * G * F;
    float denominator   = 4.0 * NdotV * NdotL + 1e-5f;
    glm::vec3 specular  = numerator / denominator;

    glm::vec3 kS = F;
    glm::vec3 kD = glm::vec3(1.0) - kS;
    kD *= (1.0 - metallic);

    atcg::BSDFEvalResult result;
    result.bsdf_value = specular + kD * diffuse_color / glm::pi<float>();

    return result;
}

extern "C" __device__ atcg::BSDFSamplingResult
__direct_callable__sample_refractivebsdf(const atcg::SurfaceInteraction& si, atcg::PCG32& rng)
{
    const atcg::RefractiveBSDFData* sbt_data =
        *reinterpret_cast<const atcg::RefractiveBSDFData**>(optixGetSbtDataPointer());

    float4 color_u          = tex2D<float4>(sbt_data->diffuse_texture, si.uv.x, si.uv.y);
    glm::vec3 diffuse_color = glm::vec3(color_u.x, color_u.y, color_u.z);


    // Determine surface parameters
    bool outsidein             = glm::dot(si.incoming_direction, si.normal) < 0;
    glm::vec3 interface_normal = outsidein ? si.normal : -si.normal;
    float eta                  = outsidein ? 1.0f / sbt_data->ior : sbt_data->ior;

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

extern "C" __device__ atcg::BSDFEvalResult __direct_callable__eval_refractivebsdf(const atcg::SurfaceInteraction& si,
                                                                                  const glm::vec3& incoming_dir)
{
    // TODO: Return empty result for now because most of the time a randomly sampled direction will not hit the delta
    // function of a perfect reflection/transmission
    return atcg::BSDFEvalResult();
}