#pragma cuda_source_property_format = PTX

#include <iostream>
#include <Pathtracing/PathtracingPlatform.h>
#include <Pathtracing/EmitterModels.cuh>

#include <Pathtracing/Stubs.h>

// TODO:
ATCG_INLINE glm::vec4 texture(const atcg::textureObject& texture, const glm::vec2& uv)
{
    uint32_t image_width  = texture.width;
    uint32_t image_height = texture.height;
    glm::ivec2 pixel(uv.x * image_width, image_height - uv.y * image_height);
    pixel = glm::clamp(pixel, glm::ivec2(0), glm::ivec2(image_width - 1, image_height - 1));

    glm::vec4 color;

    uint32_t channels = texture.channels;
    bool is_hdr       = texture.is_hdr;
    if(is_hdr)
    {
        if(channels == 4)
        {
            color = glm::vec4(((float*)texture.data)[0 + pixel.x * channels + pixel.y * channels * image_width],
                              ((float*)texture.data)[1 + pixel.x * channels + pixel.y * channels * image_width],
                              ((float*)texture.data)[2 + pixel.x * channels + pixel.y * channels * image_width],
                              ((float*)texture.data)[3 + pixel.x * channels + pixel.y * channels * image_width]);
        }
        else if(channels == 3)
        {
            color = glm::vec4(((float*)texture.data)[0 + pixel.x * channels + pixel.y * channels * image_width],
                              ((float*)texture.data)[1 + pixel.x * channels + pixel.y * channels * image_width],
                              ((float*)texture.data)[2 + pixel.x * channels + pixel.y * channels * image_width],
                              1.0f);
        }
        else if(channels == 2)
        {
            color = glm::vec4(((float*)texture.data)[0 + pixel.x * channels + pixel.y * channels * image_width],
                              ((float*)texture.data)[1 + pixel.x * channels + pixel.y * channels * image_width],
                              0.0f,
                              1.0f);
        }
        else
        {
            color = glm::vec4(((float*)texture.data)[0 + pixel.x * channels + pixel.y * channels * image_width],
                              0.0f,
                              0.0f,
                              1.0f);
        }
    }
    else
    {
        if(channels == 4)
        {
            glm::u8vec4 val = glm::u8vec4(((uint8_t*)texture.data)[0],
                                          ((uint8_t*)texture.data)[1],
                                          ((uint8_t*)texture.data)[2],
                                          ((uint8_t*)texture.data)[3]);
            color           = glm::vec4((float)val.x, (float)val.y, (float)val.z, (float)val.w) / 255.0f;
        }
        else if(channels == 3)
        {
            glm::u8vec4 val =
                glm::u8vec4(((uint8_t*)texture.data)[0 + pixel.x * channels + pixel.y * channels * image_width],
                            ((uint8_t*)texture.data)[1 + pixel.x * channels + pixel.y * channels * image_width],
                            ((uint8_t*)texture.data)[2 + pixel.x * channels + pixel.y * channels * image_width],
                            255);
            color = glm::vec4((float)val.x, (float)val.y, (float)val.z, (float)val.w) / 255.0f;
        }
        else if(channels == 2)
        {
            glm::u8vec4 val =
                glm::u8vec4(((uint8_t*)texture.data)[0 + pixel.x * channels + pixel.y * channels * image_width],
                            ((uint8_t*)texture.data)[1 + pixel.x * channels + pixel.y * channels * image_width],
                            0,
                            255);
            color = glm::vec4((float)val.x, (float)val.y, (float)val.z, (float)val.w) / 255.0f;
        }
        else
        {
            uint8_t val = ((uint8_t*)texture.data)[0 + pixel.x * channels + pixel.y * channels * image_width];
            color       = glm::vec4((float)val, 0.0f, 0.0f, 255.0f) / 255.0f;
        }
    }

    return color;
}

extern "C" ATCG_RT_EXPORT atcg::EmitterSamplingResult
__direct_callable__sample_meshemitter(const atcg::SurfaceInteraction& si, atcg::PCG32& rng)
{
    // TODO:
    atcg::MeshEmitterData* sbt_data = *reinterpret_cast<atcg::MeshEmitterData**>(atcg::g_function_memory_map["__direct_"
                                                                                                             "callable_"
                                                                                                             "_sample_"
                                                                                                             "meshemitt"
                                                                                                             "er"]);
    atcg::EmitterSamplingResult result = atcg::sampleMeshEmitter(si,
                                                                 sbt_data->mesh_cdf,
                                                                 (glm::vec3*)sbt_data->positions,
                                                                 (glm::vec3*)sbt_data->uvs,
                                                                 (glm::u32vec3*)sbt_data->faces,
                                                                 sbt_data->num_faces,
                                                                 sbt_data->total_area,
                                                                 sbt_data->local_to_world,
                                                                 sbt_data->world_to_local,
                                                                 rng);

    glm::vec3 emissive_color = texture(sbt_data->emissive_texture, si.uv);

    result.radiance_weight_at_receiver = sbt_data->emitter_scaling * emissive_color / result.sampling_pdf;

    return result;
}

extern "C" ATCG_RT_EXPORT glm::vec3 __direct_callable__eval_meshemitter(const atcg::SurfaceInteraction& si)
{
    // TODO
    atcg::MeshEmitterData* sbt_data = *reinterpret_cast<atcg::MeshEmitterData**>(atcg::g_function_memory_map["__direct_"
                                                                                                             "callable_"
                                                                                                             "_eval_"
                                                                                                             "meshemitt"
                                                                                                             "er"]);
    glm::vec3 emissive_color        = texture(sbt_data->emissive_texture, si.uv);

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
    // TODO:
    atcg::EnvironmentEmitterData* sbt_data =
        *reinterpret_cast<atcg::EnvironmentEmitterData**>(atcg::g_function_memory_map["__direct_callable__sample_"
                                                                                      "environmentemitter"]);
    auto result                        = sampleEnvironmentEmitter(si, rng);
    result.radiance_weight_at_receiver = texture(sbt_data->environment_texture, result.uvs) / result.sampling_pdf;

    return result;
}

extern "C" ATCG_RT_EXPORT glm::vec3 __direct_callable__eval_environmentemitter(const atcg::SurfaceInteraction& si)
{
    // TODO:
    atcg::EnvironmentEmitterData* sbt_data =
        *reinterpret_cast<atcg::EnvironmentEmitterData**>(atcg::g_function_memory_map["__direct_callable__eval_"
                                                                                      "environmentemitter"]);
    auto uv = evalEnvironmentEmitter(si);

    return texture(sbt_data->environment_texture, uv);
}

extern "C" ATCG_RT_EXPORT float __direct_callable__evalpdf_environmentemitter(const atcg::SurfaceInteraction& last_si,
                                                                              const atcg::SurfaceInteraction& si)
{
    return evalEnvironmentEmitterSamplingPdf(last_si, si);
}