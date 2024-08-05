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

    return atcg::samplePBR(si, diffuse_color, metallic_color, metallic, roughness, rng);
}

extern "C" __device__ atcg::BSDFEvalResult __direct_callable__eval_pbrbsdf(const atcg::SurfaceInteraction& si,
                                                                           const glm::vec3& outgoing_dir)
{
    const atcg::PBRBSDFData* sbt_data = *reinterpret_cast<const atcg::PBRBSDFData**>(optixGetSbtDataPointer());
    atcg::BSDFEvalResult result;

    float4 color_u          = tex2D<float4>(sbt_data->diffuse_texture, si.uv.x, si.uv.y);
    glm::vec3 diffuse_color = glm::vec3(color_u.x, color_u.y, color_u.z);
    float metallic          = tex2D<float>(sbt_data->metallic_texture, si.uv.x, si.uv.y);
    float roughness         = tex2D<float>(sbt_data->roughness_texture, si.uv.x, si.uv.y);
    roughness = glm::max(roughness * roughness, 1e-3f);    // In the real time shaders, roughness is squared

    glm::vec3 metallic_color = (1.0f - metallic) * glm::vec3(0.04f) + metallic * diffuse_color;

    return atcg::evalPBR(si, outgoing_dir, diffuse_color, metallic_color, roughness, metallic);
}

extern "C" __device__ atcg::BSDFSamplingResult
__direct_callable__sample_refractivebsdf(const atcg::SurfaceInteraction& si, atcg::PCG32& rng)
{
    const atcg::RefractiveBSDFData* sbt_data =
        *reinterpret_cast<const atcg::RefractiveBSDFData**>(optixGetSbtDataPointer());

    float4 color_u          = tex2D<float4>(sbt_data->diffuse_texture, si.uv.x, si.uv.y);
    glm::vec3 diffuse_color = glm::vec3(color_u.x, color_u.y, color_u.z);

    return atcg::sampleRefractive(si, diffuse_color, sbt_data->ior, rng);
}

extern "C" __device__ atcg::BSDFEvalResult __direct_callable__eval_refractivebsdf(const atcg::SurfaceInteraction& si,
                                                                                  const glm::vec3& incoming_dir)
{
    return atcg::evalRefractive(si, incoming_dir);
}