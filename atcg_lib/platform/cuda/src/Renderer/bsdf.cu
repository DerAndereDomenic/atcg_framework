#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>
#include <Math/SurfaceInteraciton.h>

#include <Renderer/BSDFModels.cuh>

extern "C" __device__ atcg::BSDFSamplingResult __direct_callable__sample_bsdf(const atcg::SurfaceInteraction& si,
                                                                              atcg::PCG32& rng)
{
    const atcg::PBRBSDFData* sbt_data = *reinterpret_cast<const atcg::PBRBSDFData**>(optixGetSbtDataPointer());

    uchar4 color_u = tex1Dfetch<uchar4>(sbt_data->diffuse_texture, 0);
    glm::vec3 diffuse_color =
        glm::vec3((float)color_u.x / 255.0f, (float)color_u.y / 255.0f, (float)color_u.z / 255.0f);
    float metallic  = tex1Dfetch<float>(sbt_data->metallic_texture, 0);
    float roughness = tex1Dfetch<float>(sbt_data->roughness_texture, 0);

    glm::vec3 metallic_color = (1.0f - metallic) * glm::vec3(0.04f) + metallic * diffuse_color;

    atcg::BSDFSamplingResult result = atcg::sampleGGX(si, diffuse_color, metallic_color, metallic, roughness, rng);

    return result;
}