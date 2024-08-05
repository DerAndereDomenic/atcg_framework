#include <DataStructure/TorchUtils.h>

namespace atcg
{
glm::vec4 texture(const torch::Tensor& image, const glm::vec2& uv)
{
    TORCH_CHECK_EQ(image.ndimension(), 3);
    TORCH_CHECK_GE(uv.x, 0.0f);
    TORCH_CHECK_GE(uv.y, 0.0f);
    TORCH_CHECK_LE(uv.x, 1.0f);
    TORCH_CHECK_LE(uv.y, 1.0f);

    uint32_t image_width  = image.size(1);
    uint32_t image_height = image.size(0);
    glm::ivec2 pixel(uv.x * image_width, image_height - uv.y * image_height);
    pixel = glm::clamp(pixel, glm::ivec2(0), glm::ivec2(image_width - 1, image_height - 1));

    glm::vec4 color;

    uint32_t channels = image.size(2);
    bool is_hdr       = image.scalar_type() == torch::kFloat32;
    if(is_hdr)
    {
        if(channels == 4)
        {
            color = glm::vec4(image.index({pixel.y, pixel.x, 0}).item<float>(),
                              image.index({pixel.y, pixel.x, 1}).item<float>(),
                              image.index({pixel.y, pixel.x, 2}).item<float>(),
                              image.index({pixel.y, pixel.x, 3}).item<float>());
        }
        else if(channels == 3)
        {
            color = glm::vec4(image.index({pixel.y, pixel.x, 0}).item<float>(),
                              image.index({pixel.y, pixel.x, 1}).item<float>(),
                              image.index({pixel.y, pixel.x, 2}).item<float>(),
                              1.0f);
        }
        else if(channels == 2)
        {
            color = glm::vec4(image.index({pixel.y, pixel.x, 0}).item<float>(),
                              image.index({pixel.y, pixel.x, 1}).item<float>(),
                              0.0f,
                              1.0f);
        }
        else
        {
            color = glm::vec4(image.index({pixel.y, pixel.x, 0}).item<float>(), 0.0f, 0.0f, 1.0f);
        }
    }
    else
    {
        if(channels == 4)
        {
            glm::u8vec4 val = glm::u8vec4(image.index({pixel.y, pixel.x, 0}).item<uint8_t>(),
                                          image.index({pixel.y, pixel.x, 1}).item<uint8_t>(),
                                          image.index({pixel.y, pixel.x, 2}).item<uint8_t>(),
                                          image.index({pixel.y, pixel.x, 3}).item<uint8_t>());
            color           = glm::vec4((float)val.x, (float)val.y, (float)val.z, (float)val.w) / 255.0f;
        }
        else if(channels == 3)
        {
            glm::u8vec4 val = glm::u8vec4(image.index({pixel.y, pixel.x, 0}).item<uint8_t>(),
                                          image.index({pixel.y, pixel.x, 1}).item<uint8_t>(),
                                          image.index({pixel.y, pixel.x, 2}).item<uint8_t>(),
                                          255);
            color           = glm::vec4((float)val.x, (float)val.y, (float)val.z, (float)val.w) / 255.0f;
        }
        else if(channels == 2)
        {
            glm::u8vec4 val = glm::u8vec4(image.index({pixel.y, pixel.x, 0}).item<uint8_t>(),
                                          image.index({pixel.y, pixel.x, 1}).item<uint8_t>(),
                                          0,
                                          255);
            color           = glm::vec4((float)val.x, (float)val.y, (float)val.z, (float)val.w) / 255.0f;
        }
        else
        {
            uint8_t val = image.index({pixel.y, pixel.x, 0}).item<uint8_t>();
            color       = glm::vec4((float)val, 0.0f, 0.0f, 255.0f) / 255.0f;
        }
    }

    return color;
}
}    // namespace atcg