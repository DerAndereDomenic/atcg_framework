#pragma once

#include <Core/API.h>
#include <Core/CUDA.h>

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

ATCG_API const char* textureFormatToString(TextureFormat format);

ATCG_API TextureFormat stringToTextureFormat(const char* str);

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

ATCG_API const char* textureWrapModeToString(TextureWrapMode mode);

ATCG_API TextureWrapMode stringToTextureWrapMode(const char* str);

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

ATCG_API const char* textureFilterModeToString(TextureFilterMode mode);

ATCG_API TextureFilterMode stringToTextureFilterMode(const char* str);

/**
 * @brief The texture sampler.
 */
struct TextureSamplerSpecification
{
    TextureWrapMode wrap_mode     = TextureWrapMode::REPEAT;
    TextureFilterMode filter_mode = TextureFilterMode::LINEAR;
    bool mip_map                  = false;
};

struct ATCG_API TextureSpecification
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
    TextureSpecification(uint32_t width,
                         uint32_t height,
                         uint32_t depth,
                         TextureFormat format,
                         TextureSamplerSpecification sampler)
        : format(format),
          sampler(sampler),
          width(width),
          height(height),
          depth(depth)
    {
    }

    TextureFormat format                = TextureFormat::RGBA;
    TextureSamplerSpecification sampler = {};
    uint32_t width                      = 0;
    uint32_t height                     = 0;
    uint32_t depth                      = 0;
    uint32_t num_samples                = 1;

    ATCG_INLINE ATCG_HOST_DEVICE std::size_t pixelSize() const
    {
        switch(format)
        {
            case TextureFormat::RG:
            {
                return 2 * sizeof(uint8_t);
            }
            case TextureFormat::RGB:
            {
                return 3 * sizeof(uint8_t);
            }
            case TextureFormat::RGBA:
            {
                return 4 * sizeof(uint8_t);
            }
            case TextureFormat::RGFLOAT:
            {
                return 2 * sizeof(float);
            }
            case TextureFormat::RGBFLOAT:
            {
                return 3 * sizeof(float);
            }
            case TextureFormat::RGBAFLOAT:
            {
                return 4 * sizeof(float);
            }
            case TextureFormat::RINT:
            {
                return sizeof(uint32_t);
            }
            case TextureFormat::RINT8:
            {
                return sizeof(uint8_t);
            }
            case TextureFormat::RFLOAT:
            {
                return sizeof(float);
            }
            case TextureFormat::DEPTH:
            {
                return sizeof(float);
            }
            default:
            {
                ATCG_ERROR("Unknown TextureFormat {0}", (int)format);
                return 0;
            }
        }
    }

    ATCG_INLINE ATCG_HOST_DEVICE std::size_t channelSize() const
    {
        switch(format)
        {
            case TextureFormat::RG:
            {
                return sizeof(uint8_t);
            }
            case TextureFormat::RGB:
            {
                return sizeof(uint8_t);
            }
            case TextureFormat::RGBA:
            {
                return sizeof(uint8_t);
            }
            case TextureFormat::RGFLOAT:
            {
                return sizeof(float);
            }
            case TextureFormat::RGBFLOAT:
            {
                return sizeof(float);
            }
            case TextureFormat::RGBAFLOAT:
            {
                return sizeof(float);
            }
            case TextureFormat::RINT:
            {
                return sizeof(uint32_t);
            }
            case TextureFormat::RINT8:
            {
                return sizeof(uint8_t);
            }
            case TextureFormat::RFLOAT:
            {
                return sizeof(float);
            }
            case TextureFormat::DEPTH:
            {
                return sizeof(float);
            }
            default:
            {
                ATCG_ERROR("Unknown TextureFormat {0}", (int)format);
                return 0;
            }
        }
    }

    ATCG_INLINE ATCG_HOST_DEVICE uint32_t numChannels() const
    {
        switch(format)
        {
            case TextureFormat::RG:
            {
                return 2;
            }
            case TextureFormat::RGB:
            {
                return 3;
            }
            case TextureFormat::RGBA:
            {
                return 4;
            }
            case TextureFormat::RGFLOAT:
            {
                return 2;
            }
            case TextureFormat::RGBFLOAT:
            {
                return 3;
            }
            case TextureFormat::RGBAFLOAT:
            {
                return 4;
            }
            case TextureFormat::RINT:
            {
                return 1;
            }
            case TextureFormat::RINT8:
            {
                return 1;
            }
            case TextureFormat::RFLOAT:
            {
                return 1;
            }
            case TextureFormat::DEPTH:
            {
                return 1;
            }
            default:
            {
                ATCG_ERROR("Unknown TextureFormat {0}", (int)format);
                return 0;
            }
        }
    }

    ATCG_INLINE ATCG_HOST_DEVICE bool isFloat() const
    {
        switch(format)
        {
            case TextureFormat::RFLOAT:
            case TextureFormat::RGFLOAT:
            case TextureFormat::RGBFLOAT:
            case TextureFormat::RGBAFLOAT:
            case TextureFormat::DEPTH:
                return true;
            default:
                return false;
        }

        return false;
    }

    ATCG_INLINE ATCG_HOST_DEVICE bool isInt() const { return format == TextureFormat::RINT; }
};
}    // namespace atcg