#pragma cuda_source_property_format = PTX

#include <iostream>
#include <Pathtracing/PathtracingPlatform.h>
#include <Pathtracing/BSDFModels.cuh>
#include <Pathtracing/Stubs.h>

// TODO: Has to be removed at some point
#include <DataStructure/TorchUtils.h>

extern "C" ATCG_RT_EXPORT atcg::BSDFSamplingResult __direct_callable__sample_pbrbsdf(const atcg::SurfaceInteraction& si,
                                                                                     atcg::PCG32& rng)
{
    atcg::PBRBSDFData* sbt_data = *reinterpret_cast<atcg::PBRBSDFData**>(atcg::g_function_memory_map["__direct_"
                                                                                                     "callable__"
                                                                                                     "sample_pbrbsdf"]);

    glm::vec3 diffuse_color = atcg::texture(sbt_data->diffuse_texture, si.uv);
    float metallic          = atcg::texture(sbt_data->metallic_texture, si.uv).x;
    float roughness         = atcg::texture(sbt_data->roughness_texture, si.uv).x;

    glm::vec3 metallic_color = (1.0f - metallic) * glm::vec3(0.04) + metallic * diffuse_color;

    return samplePBR(si, diffuse_color, metallic_color, metallic, roughness, rng);
}

extern "C" ATCG_RT_EXPORT atcg::BSDFEvalResult __direct_callable__eval_pbrbsdf(const atcg::SurfaceInteraction& si,
                                                                               const glm::vec3& outgoing_dir)
{
    atcg::PBRBSDFData* sbt_data = *reinterpret_cast<atcg::PBRBSDFData**>(atcg::g_function_memory_map["__direct_"
                                                                                                     "callable__eval_"
                                                                                                     "pbrbsdf"]);
    atcg::BSDFEvalResult result;

    glm::vec3 diffuse_color = atcg::texture(sbt_data->diffuse_texture, si.uv);
    float metallic          = atcg::texture(sbt_data->metallic_texture, si.uv).x;
    float roughness         = atcg::texture(sbt_data->roughness_texture, si.uv).x;
    roughness = glm::max(roughness * roughness, 1e-3f);    // In the real time shaders, roughness is squared

    glm::vec3 metallic_color = (1.0f - metallic) * glm::vec3(0.04f) + metallic * diffuse_color;

    return evalPBR(si, outgoing_dir, diffuse_color, metallic_color, roughness, metallic);
}

extern "C" ATCG_RT_EXPORT atcg::BSDFSamplingResult
__direct_callable__sample_refractivebsdf(const atcg::SurfaceInteraction& si, atcg::PCG32& rng)
{
    atcg::RefractiveBSDFData* sbt_data =
        *reinterpret_cast<atcg::RefractiveBSDFData**>(atcg::g_function_memory_map["__direct_callable__sample_"
                                                                                  "refractivebsdf"]);
    glm::vec3 diffuse_color = atcg::texture(sbt_data->diffuse_texture, si.uv);

    return sampleRefractive(si, diffuse_color, sbt_data->ior, rng);
}

extern "C" ATCG_RT_EXPORT atcg::BSDFEvalResult
__direct_callable__eval_refractivebsdf(const atcg::SurfaceInteraction& si, const glm::vec3& incoming_dir)
{
    return evalRefractive(si, incoming_dir);
}