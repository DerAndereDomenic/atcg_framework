#pragma once

namespace atcg
{
/**
 * @brief The format of the texture.
 */
enum class TextureFormat
{
    // RG unsigned byte color texture.
    RG = 0,
    // RGB unsigned byte color texture.
    RGB,
    // RGBA unsigned byte color texture.
    RGBA,
    // RG float color texture.
    RGFLOAT,
    // RGB float color texture.
    RGBFLOAT,
    // RGBA float color texture.
    RGBAFLOAT,
    // Red channel unsigned int texture.
    RINT,
    // Red channel byte texture.
    RINT8,
    // Red channel float 32 texture.
    RFLOAT,
    // Depth texture.
    DEPTH,
    _NUM_FORMATS
};

const char* textureFormatToString(TextureFormat format);

TextureFormat stringToTextureFormat(const char* str);

/**
 * @brief The texture wrap mode.
 */
enum class TextureWrapMode
{
    // Extend the texture by the border pixel values
    CLAMP_TO_EDGE,
    // Repeat the texture
    REPEAT,
    // Border
    BORDER
};

const char* textureWrapModeToString(TextureWrapMode mode);

TextureWrapMode stringToTextureWrapMode(const char* str);

/**
 * @brief The texture filter mode.
 */
enum class TextureFilterMode
{
    // Nearest neighbor filter.
    NEAREST,
    // Linear interpolation.
    LINEAR,
    // Trilinear interpolation using mipmaps (Needs to have a TextureSampler with mip_map = true)
    MIPMAP_LINEAR
};

const char* textureFilterModeToString(TextureFilterMode mode);

TextureFilterMode stringToTextureFilterMode(const char* str);

/**
 * @brief The texture sampler.
 */
struct TextureSampler
{
    TextureWrapMode wrap_mode     = TextureWrapMode::REPEAT;
    TextureFilterMode filter_mode = TextureFilterMode::LINEAR;
    bool mip_map                  = false;
};

struct TextureSpecification
{
    TextureSpecification() = default;
    TextureSpecification(TextureFormat format) : format(format) {}

    TextureSpecification(uint32_t width, uint32_t height, TextureFormat format)
        : width(width),
          height(height),
          format(format)
    {
    }

    TextureSpecification(uint32_t width, uint32_t height, uint32_t depth, TextureFormat format)
        : width(width),
          height(height),
          depth(depth),
          format(format)
    {
    }
    TextureSpecification(uint32_t width, uint32_t height, uint32_t depth, TextureFormat format, TextureSampler sampler)
        : format(format),
          sampler(sampler),
          width(width),
          height(height),
          depth(depth)
    {
    }

    TextureFormat format   = TextureFormat::RGBA;
    TextureSampler sampler = {};
    uint32_t width         = 0;
    uint32_t height        = 0;
    uint32_t depth         = 0;
};
}    // namespace atcg