#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>
#include <Pathtracing/SurfaceInteraction.h>

#include <Pathtracing/Emitter/EmitterModels.cuh>
#include <Pathtracing/BSDF/BSDFModels.cuh>

extern "C" __device__ atcg::EmitterSamplingResult
__direct_callable__sample_meshemitter(const atcg::SurfaceInteraction& si, atcg::PCG32& rng)
{
    const atcg::MeshEmitterData* sbt_data = *reinterpret_cast<const atcg::MeshEmitterData**>(optixGetSbtDataPointer());
    atcg::EmitterSamplingResult result    = atcg::sampleMeshEmitter(si,
                                                                 sbt_data->mesh_cdf,
                                                                 sbt_data->positions,
                                                                 sbt_data->uvs,
                                                                 sbt_data->faces,
                                                                 sbt_data->num_faces,
                                                                 sbt_data->total_area,
                                                                 sbt_data->local_to_world,
                                                                 sbt_data->world_to_local,
                                                                 rng);

    float4 color_u           = tex2D<float4>(sbt_data->emissive_texture, result.uvs.x, result.uvs.y);
    glm::vec3 emissive_color = glm::vec3(color_u.x, color_u.y, color_u.z);

    result.radiance_weight_at_receiver = sbt_data->emitter_scaling * emissive_color / result.sampling_pdf;

    return result;
}

extern "C" __device__ glm::vec3 __direct_callable__eval_meshemitter(const atcg::SurfaceInteraction& si)
{
    const atcg::MeshEmitterData* sbt_data = *reinterpret_cast<const atcg::MeshEmitterData**>(optixGetSbtDataPointer());

    float4 color_u           = tex2D<float4>(sbt_data->emissive_texture, si.uv.x, si.uv.y);
    glm::vec3 emissive_color = glm::vec3(color_u.x, color_u.y, color_u.z);

    return atcg::evalMeshEmitter(emissive_color, sbt_data->emitter_scaling);
}

extern "C" __device__ float __direct_callable__evalpdf_meshemitter(const atcg::SurfaceInteraction& last_si,
                                                                   const atcg::SurfaceInteraction& si)
{
    const atcg::MeshEmitterData* sbt_data = *reinterpret_cast<const atcg::MeshEmitterData**>(optixGetSbtDataPointer());
    // We can assume that outgoing ray dir actually intersects the light source.

    return atcg::evalMeshEmitterPDF(last_si, sbt_data->total_area, si);
}

extern "C" __device__ atcg::EmitterSamplingResult
__direct_callable__sample_environmentemitter(const atcg::SurfaceInteraction& si, atcg::PCG32& rng)
{
    const atcg::EnvironmentEmitterData* sbt_data =
        *reinterpret_cast<const atcg::EnvironmentEmitterData**>(optixGetSbtDataPointer());

    atcg::EmitterSamplingResult result = atcg::sampleEnvironmentEmitter(si, rng);

    float4 color = tex2D<float4>(sbt_data->environment_texture, result.uvs.x, 1.0f - result.uvs.y);

    result.distance_to_light           = std::numeric_limits<float>::infinity();
    result.sampling_pdf                = result.sampling_pdf;
    result.radiance_weight_at_receiver = glm::vec3(color.x, color.y, color.z) / result.sampling_pdf;

    return result;
}

extern "C" __device__ glm::vec3 __direct_callable__eval_environmentemitter(const atcg::SurfaceInteraction& si)
{
    const atcg::EnvironmentEmitterData* sbt_data =
        *reinterpret_cast<const atcg::EnvironmentEmitterData**>(optixGetSbtDataPointer());

    glm::vec2 uv = atcg::evalEnvironmentEmitter(si);

    float4 color = tex2D<float4>(sbt_data->environment_texture, uv.x, 1.0f - uv.y);

    return glm::vec3(color.x, color.y, color.z);
}

extern "C" __device__ float __direct_callable__evalpdf_environmentemitter(const atcg::SurfaceInteraction& last_si,
                                                                          const atcg::SurfaceInteraction& si)
{
    const atcg::EnvironmentEmitterData* sbt_data =
        *reinterpret_cast<const atcg::EnvironmentEmitterData**>(optixGetSbtDataPointer());
    // We can assume that outgoing ray dir actually intersects the light source.

    return atcg::evalEnvironmentEmitterSamplingPdf(last_si, si);
}
