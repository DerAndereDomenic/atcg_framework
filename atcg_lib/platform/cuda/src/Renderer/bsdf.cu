#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>
#include <Math/SurfaceInteraciton.h>

#include <Renderer/BSDFModels.cuh>

extern "C" __device__ atcg::BSDFSamplingResult __direct_callable__sample_bsdf(const atcg::SurfaceInteraction& si,
                                                                              atcg::PCG32& rng)
{
    const atcg::PBRBSDFData* sbt_data = *reinterpret_cast<const atcg::PBRBSDFData**>(optixGetSbtDataPointer());

    float4 color_u          = tex2D<float4>(sbt_data->diffuse_texture, si.uv.x, si.uv.y);
    glm::vec3 diffuse_color = glm::vec3(color_u.x, color_u.y, color_u.z);
    float metallic          = tex2D<float>(sbt_data->metallic_texture, si.uv.x, si.uv.y);
    float roughness         = tex2D<float>(sbt_data->roughness_texture, si.uv.x, si.uv.y);

    glm::vec3 metallic_color = (1.0f - metallic) * glm::vec3(0.04f) + metallic * diffuse_color;

    atcg::BSDFSamplingResult result = atcg::sampleGGX(si, diffuse_color, metallic_color, metallic, roughness, rng);

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