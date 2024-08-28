#pragma cuda_source_property_format = PTX

#include <iostream>
#include <Pathtracing/PathtracingPlatform.h>
#include <Pathtracing/BSDFModels.cuh>
#include <Pathtracing/Stubs.h>

// TODO:
ATCG_INLINE glm::vec4 texture(const atcg::textureArray& texture, const glm::vec2& uv)
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

extern "C" ATCG_RT_EXPORT atcg::BSDFSamplingResult __direct_callable__sample_pbrbsdf(const atcg::SurfaceInteraction& si,
                                                                                     atcg::PCG32& rng)
{
    // TODO: Method to retreive sbt data
    atcg::PBRBSDFData* sbt_data = *reinterpret_cast<atcg::PBRBSDFData**>(atcg::g_function_memory_map["__direct_"
                                                                                                     "callable__"
                                                                                                     "sample_pbrbsdf"]);

    glm::vec3 diffuse_color = texture(sbt_data->diffuse_texture, si.uv);
    float metallic          = texture(sbt_data->metallic_texture, si.uv).x;
    float roughness         = texture(sbt_data->roughness_texture, si.uv).x;

    glm::vec3 metallic_color = (1.0f - metallic) * glm::vec3(0.04) + metallic * diffuse_color;

    return samplePBR(si, diffuse_color, metallic_color, metallic, roughness, rng);
}

extern "C" ATCG_RT_EXPORT atcg::BSDFEvalResult __direct_callable__eval_pbrbsdf(const atcg::SurfaceInteraction& si,
                                                                               const glm::vec3& outgoing_dir)
{
    // TODO: Method to retreive sbt data
    atcg::PBRBSDFData* sbt_data = *reinterpret_cast<atcg::PBRBSDFData**>(atcg::g_function_memory_map["__direct_"
                                                                                                     "callable__eval_"
                                                                                                     "pbrbsdf"]);
    atcg::BSDFEvalResult result;

    glm::vec3 diffuse_color = texture(sbt_data->diffuse_texture, si.uv);
    float metallic          = texture(sbt_data->metallic_texture, si.uv).x;
    float roughness         = texture(sbt_data->roughness_texture, si.uv).x;
    roughness = glm::max(roughness * roughness, 1e-3f);    // In the real time shaders, roughness is squared

    glm::vec3 metallic_color = (1.0f - metallic) * glm::vec3(0.04f) + metallic * diffuse_color;

    return evalPBR(si, outgoing_dir, diffuse_color, metallic_color, roughness, metallic);
}

extern "C" ATCG_RT_EXPORT atcg::BSDFSamplingResult
__direct_callable__sample_refractivebsdf(const atcg::SurfaceInteraction& si, atcg::PCG32& rng)
{
    // TODO: Method to retreive sbt data
    atcg::RefractiveBSDFData* sbt_data =
        *reinterpret_cast<atcg::RefractiveBSDFData**>(atcg::g_function_memory_map["__direct_callable__sample_"
                                                                                  "refractivebsdf"]);
    glm::vec3 diffuse_color = texture(sbt_data->diffuse_texture, si.uv);

    return sampleRefractive(si, diffuse_color, sbt_data->ior, rng);
}

extern "C" ATCG_RT_EXPORT atcg::BSDFEvalResult
__direct_callable__eval_refractivebsdf(const atcg::SurfaceInteraction& si, const glm::vec3& incoming_dir)
{
    return evalRefractive(si, incoming_dir);
}