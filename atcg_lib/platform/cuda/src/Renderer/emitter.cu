#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>
#include <Math/SurfaceInteraction.h>

#include <Renderer/Emitter.cuh>

extern "C" __device__ atcg::EmitterSamplingResult
__direct_callable__sample_meshemitter(const atcg::SurfaceInteraction& si, atcg::PCG32& rng)
{
    atcg::EmitterSamplingResult result;


    return result;
}

extern "C" __device__ glm::vec3 __direct_callable__eval_meshemitter(const atcg::SurfaceInteraction& si)
{
    const atcg::MeshEmitterData* sbt_data = *reinterpret_cast<const atcg::MeshEmitterData**>(optixGetSbtDataPointer());

    float4 color_u           = tex2D<float4>(sbt_data->emissive_texture, si.uv.x, si.uv.y);
    glm::vec3 emissive_color = glm::vec3(color_u.x, color_u.y, color_u.z);

    return emissive_color * sbt_data->emitter_scaling;
}