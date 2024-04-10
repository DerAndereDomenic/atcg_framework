#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>
#include <Math/SurfaceInteraciton.h>

#include <Renderer/BSDFModels.h>

extern "C" __device__ atcg::BSDFSamplingResult __direct_callable__sample_bsdf(const atcg::SurfaceInteraction& si,
                                                                              atcg::PCG32& rng)
{
    glm::vec3 diffuse_color = glm::vec3(1);
    float metallic          = 0.7f;
    float roughness         = 0.2f;

    glm::vec3 metallic_color = (1.0f - metallic) * glm::vec3(0.04f) + metallic * diffuse_color;

    atcg::BSDFSamplingResult result = atcg::sampleGGX(si, diffuse_color, metallic_color, metallic, roughness, rng);

    return result;
}