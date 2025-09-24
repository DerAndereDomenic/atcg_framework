#include <Renderer/TextureSpecification.h>

namespace atcg
{
const char* textureFormatToString(TextureFormat format)
{
    switch(format)
    {
        case TextureFormat::RG:
            return "RG";
        case TextureFormat::RGB:
            return "RGB";
        case TextureFormat::RGBA:
            return "RGBA";
        case TextureFormat::RGFLOAT:
            return "RGFLOAT";
        case TextureFormat::RGBFLOAT:
            return "RGBFLOAT";
        case TextureFormat::RGBAFLOAT:
            return "RGBAFLOAT";
        case TextureFormat::RINT:
            return "RINT";
        case TextureFormat::RINT8:
            return "RINT8";
        case TextureFormat::RFLOAT:
            return "RFLOAT";
        case TextureFormat::DEPTH:
            return "DEPTH";
        default:
            return "UNKNOWN";
    }
}

TextureFormat stringToTextureFormat(const char* str)
{
    if(!str) return TextureFormat::DEPTH;    // fallback, could also throw

    if(std::strcmp(str, "RG") == 0) return TextureFormat::RG;
    if(std::strcmp(str, "RGB") == 0) return TextureFormat::RGB;
    if(std::strcmp(str, "RGBA") == 0) return TextureFormat::RGBA;
    if(std::strcmp(str, "RGFLOAT") == 0) return TextureFormat::RGFLOAT;
    if(std::strcmp(str, "RGBFLOAT") == 0) return TextureFormat::RGBFLOAT;
    if(std::strcmp(str, "RGBAFLOAT") == 0) return TextureFormat::RGBAFLOAT;
    if(std::strcmp(str, "RINT") == 0) return TextureFormat::RINT;
    if(std::strcmp(str, "RINT8") == 0) return TextureFormat::RINT8;
    if(std::strcmp(str, "RFLOAT") == 0) return TextureFormat::RFLOAT;
    if(std::strcmp(str, "DEPTH") == 0) return TextureFormat::DEPTH;

    // fallback
    return TextureFormat::RGB;    // or throw an exception if invalid input
}

const char* textureWrapModeToString(TextureWrapMode mode)
{
    switch(mode)
    {
        case TextureWrapMode::CLAMP_TO_EDGE:
            return "CLAMP_TO_EDGE";
        case TextureWrapMode::REPEAT:
            return "REPEAT";
        case TextureWrapMode::BORDER:
            return "BORDER";
        default:
            return "UNKNOWN";
    }
}

TextureWrapMode stringToTextureWrapMode(const char* str)
{
    if(std::strcmp(str, "CLAMP_TO_EDGE") == 0)
        return TextureWrapMode::CLAMP_TO_EDGE;
    else if(std::strcmp(str, "REPEAT") == 0)
        return TextureWrapMode::REPEAT;
    else if(std::strcmp(str, "BORDER") == 0)
        return TextureWrapMode::BORDER;
    else
        throw std::invalid_argument("Invalid TextureWrapMode string");
}

const char* textureFilterModeToString(TextureFilterMode mode)
{
    switch(mode)
    {
        case TextureFilterMode::NEAREST:
            return "NEAREST";
        case TextureFilterMode::LINEAR:
            return "LINEAR";
        case TextureFilterMode::MIPMAP_LINEAR:
            return "MIPMAP_LINEAR";
        default:
            return "UNKNOWN";
    }
}

TextureFilterMode stringToTextureFilterMode(const char* str)
{
    if(std::strcmp(str, "NEAREST") == 0)
        return TextureFilterMode::NEAREST;
    else if(std::strcmp(str, "LINEAR") == 0)
        return TextureFilterMode::LINEAR;
    else if(std::strcmp(str, "MIPMAP_LINEAR") == 0)
        return TextureFilterMode::MIPMAP_LINEAR;
    else
        throw std::invalid_argument("Invalid TextureFilterMode string");
}
}    // namespace atcg