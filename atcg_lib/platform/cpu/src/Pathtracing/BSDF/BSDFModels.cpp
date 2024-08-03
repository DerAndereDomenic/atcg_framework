#include <Pathtracing/BSDF/BSDFModels.h>

namespace atcg
{

namespace detail
{
glm::vec3 read_image(const torch::Tensor& image, const glm::vec2& uv)
{
    glm::vec4 result(0);

    uint32_t image_width  = image.size(1);
    uint32_t image_height = image.size(0);
    glm::ivec2 pixel(uv.x * image_width, image_height - uv.y * image_height);
    pixel = glm::clamp(pixel, glm::ivec2(0), glm::ivec2(image_width - 1, image_height - 1));

    glm::vec3 color;

    uint32_t channels = image.size(2);
    bool is_hdr       = image.scalar_type() == torch::kFloat32;
    if(is_hdr)
    {
        if(channels == 4 || channels == 3)
        {
            color = glm::vec3(image.index({pixel.y, pixel.x, 0}).item<float>(),
                              image.index({pixel.y, pixel.x, 1}).item<float>(),
                              image.index({pixel.y, pixel.x, 2}).item<float>());
        }
        else
        {
            color = glm::vec3(image.index({pixel.y, pixel.x, 0}).item<float>());
        }
    }
    else
    {
        if(channels == 4 || channels == 3)
        {
            glm::u8vec3 val = glm::u8vec3(image.index({pixel.y, pixel.x, 0}).item<uint8_t>(),
                                          image.index({pixel.y, pixel.x, 1}).item<uint8_t>(),
                                          image.index({pixel.y, pixel.x, 2}).item<uint8_t>());
            color           = glm::vec3((float)val.x, (float)val.y, (float)val.z) / 255.0f;
        }
        else
        {
            uint8_t val = image.index({pixel.y, pixel.x, 0}).item<uint8_t>();
            color       = glm::vec3((float)val) / 255.0f;
        }
    }

    return color;
}
}    // namespace detail

PBRBSDF::PBRBSDF(const Material& material)
{
    _diffuse_image   = material.getDiffuseTexture()->getData(atcg::CPU);
    _roughness_image = material.getRoughnessTexture()->getData(atcg::CPU);
    _metallic_image  = material.getMetallicTexture()->getData(atcg::CPU);
}

PBRBSDF::~PBRBSDF() {}

BSDFSamplingResult PBRBSDF::sampleBSDF(const SurfaceInteraction& si, PCG32& rng) const
{
    glm::vec3 diffuse_color = detail::read_image(_diffuse_image, si.uv);
    float metallic          = detail::read_image(_metallic_image, si.uv).x;
    float roughness         = detail::read_image(_roughness_image, si.uv).x;

    glm::vec3 metallic_color = (1.0f - metallic) * glm::vec3(0.04) + metallic * diffuse_color;

    return sampleGGX(si, diffuse_color, metallic_color, metallic, roughness, rng);
}

BSDFEvalResult PBRBSDF::evalBSDF(const SurfaceInteraction& si, const glm::vec3& outgoing_dir)
{
    return BSDFEvalResult();
}
}    // namespace atcg