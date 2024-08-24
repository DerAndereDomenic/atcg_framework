#pragma cuda_source_property_format = PTX

#include <iostream>
#include <Pathtracing/PathtracingPlatform.h>
#include <Pathtracing/EmitterModels.cuh>

#include <Pathtracing/Stubs.h>

// TODO: has to be removed at some point
#include <DataStructure/TorchUtils.h>

extern "C" ATCG_RT_EXPORT atcg::EmitterSamplingResult
__direct_callable__sample_meshemitter(const atcg::SurfaceInteraction& si, atcg::PCG32& rng)
{
    atcg::MeshEmitterData* sbt_data = *reinterpret_cast<atcg::MeshEmitterData**>(atcg::g_function_memory_map["__direct_"
                                                                                                             "callable_"
                                                                                                             "_sample_"
                                                                                                             "meshemitt"
                                                                                                             "er"]);
    atcg::EmitterSamplingResult result = atcg::sampleMeshEmitter(si,
                                                                 sbt_data->mesh_cdf.data_ptr<float>(),
                                                                 (glm::vec3*)sbt_data->positions.data_ptr(),
                                                                 (glm::vec3*)sbt_data->uvs.data_ptr(),
                                                                 (glm::u32vec3*)sbt_data->faces.data_ptr(),
                                                                 sbt_data->faces.size(0),
                                                                 sbt_data->total_area,
                                                                 sbt_data->local_to_world,
                                                                 sbt_data->world_to_local,
                                                                 rng);

    glm::vec3 emissive_color = atcg::texture(sbt_data->emissive_texture, si.uv);

    result.radiance_weight_at_receiver = sbt_data->emitter_scaling * emissive_color / result.sampling_pdf;

    return result;
}

extern "C" ATCG_RT_EXPORT glm::vec3 __direct_callable__eval_meshemitter(const atcg::SurfaceInteraction& si)
{
    atcg::MeshEmitterData* sbt_data = *reinterpret_cast<atcg::MeshEmitterData**>(atcg::g_function_memory_map["__direct_"
                                                                                                             "callable_"
                                                                                                             "_eval_"
                                                                                                             "meshemitt"
                                                                                                             "er"]);
    glm::vec3 emissive_color        = atcg::texture(sbt_data->emissive_texture, si.uv);

    return atcg::evalMeshEmitter(emissive_color, sbt_data->emitter_scaling);
}

extern "C" ATCG_RT_EXPORT float __direct_callable__evalpdf_meshemitter(const atcg::SurfaceInteraction& last_si,
                                                                       const atcg::SurfaceInteraction& si)
{
    atcg::MeshEmitterData* sbt_data = *reinterpret_cast<atcg::MeshEmitterData**>(atcg::g_function_memory_map["__direct_"
                                                                                                             "callable_"
                                                                                                             "_evalpdf_"
                                                                                                             "meshemitt"
                                                                                                             "er"]);
    return evalMeshEmitterPDF(last_si, sbt_data->total_area, si);
}

extern "C" ATCG_RT_EXPORT atcg::EmitterSamplingResult
__direct_callable__sample_environmentemitter(const atcg::SurfaceInteraction& si, atcg::PCG32& rng)
{
    atcg::EnvironmentEmitterData* sbt_data =
        *reinterpret_cast<atcg::EnvironmentEmitterData**>(atcg::g_function_memory_map["__direct_callable__sample_"
                                                                                      "environmentemitter"]);
    auto result                        = sampleEnvironmentEmitter(si, rng);
    result.radiance_weight_at_receiver = atcg::texture(sbt_data->environment_texture, result.uvs) / result.sampling_pdf;

    return result;
}

extern "C" ATCG_RT_EXPORT glm::vec3 __direct_callable__eval_environmentemitter(const atcg::SurfaceInteraction& si)
{
    atcg::EnvironmentEmitterData* sbt_data =
        *reinterpret_cast<atcg::EnvironmentEmitterData**>(atcg::g_function_memory_map["__direct_callable__eval_"
                                                                                      "environmentemitter"]);
    auto uv = evalEnvironmentEmitter(si);

    return atcg::texture(sbt_data->environment_texture, uv);
}

extern "C" ATCG_RT_EXPORT float __direct_callable__evalpdf_environmentemitter(const atcg::SurfaceInteraction& last_si,
                                                                              const atcg::SurfaceInteraction& si)
{
    return evalEnvironmentEmitterSamplingPdf(last_si, si);
}