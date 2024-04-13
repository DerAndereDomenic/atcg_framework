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