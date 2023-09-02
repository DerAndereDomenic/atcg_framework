#include <Renderer/Texture.h>

#include <glad/glad.h>

namespace atcg
{

namespace detail
{
GLint to2GLinternalFormat(TextureFormat format)
{
    switch(format)
    {
        case TextureFormat::RGBA:
        {
            return GL_RGBA;
        }
        case TextureFormat::RINT:
        {
            return GL_R32I;
        }
        case TextureFormat::RINT8:
        {
            return GL_RED;
        }
        case TextureFormat::RFLOAT:
        {
            return GL_R32F;
        }
        case TextureFormat::DEPTH:
        {
            return GL_DEPTH_COMPONENT;
        }
        default:
        {
            ATCG_ERROR("Unknown TextureFormat {0}", (int)format);
            return -1;
        }
    }
}

GLenum toGLformat(TextureFormat format)
{
    switch(format)
    {
        case TextureFormat::RGBA:
        {
            return GL_RGBA;
        }
        case TextureFormat::RINT:
        {
            return GL_RED_INTEGER;
        }
        case TextureFormat::RINT8:
        {
            return GL_RED;
        }
        case TextureFormat::RFLOAT:
        {
            return GL_RED;
        }
        case TextureFormat::DEPTH:
        {
            return GL_DEPTH_COMPONENT;
        }
        default:
        {
            ATCG_ERROR("Unknown TextureFormat {0}", (int)format);
            return -1;
        }
    }
}

GLenum toGLtype(TextureFormat format)
{
    switch(format)
    {
        case TextureFormat::RGBA:
        {
            return GL_UNSIGNED_BYTE;
        }
        case TextureFormat::RINT:
        {
            return GL_UNSIGNED_BYTE;
        }
        case TextureFormat::RINT8:
        {
            return GL_UNSIGNED_BYTE;
        }
        case TextureFormat::RFLOAT:
        {
            return GL_FLOAT;
        }
        case TextureFormat::DEPTH:
        {
            return GL_FLOAT;
        }
        default:
        {
            ATCG_ERROR("Unknown TextureFormat {0}", (int)format);
            return -1;
        }
    }
}

GLint toGLWrapMode(TextureWrapMode wrap_mode)
{
    switch(wrap_mode)
    {
        case TextureWrapMode::CLAMP_TO_EDGE:
        {
            return GL_CLAMP_TO_EDGE;
        }
        case TextureWrapMode::REPEAT:
        {
            return GL_REPEAT;
        }
        default:
        {
            ATCG_ERROR("Unknown TextureWrapMode {0}", (int)wrap_mode);
            return -1;
        }
    }
}

GLint toGLFilterMode(TextureFilterMode filter_mode)
{
    switch(filter_mode)
    {
        case TextureFilterMode::LINEAR:
        {
            return GL_LINEAR;
        }
        case TextureFilterMode::NEAREST:
        {
            return GL_NEAREST;
        }
        default:
        {
            ATCG_ERROR("Unknown TextureFilterMode {0}", (int)filter_mode);
            return -1;
        }
    }
}
}    // namespace detail

void Texture::useForCompute(const uint32_t& slot) const
{
    glBindImageTexture(slot, _ID, 0, GL_TRUE, 0, GL_WRITE_ONLY, detail::to2GLinternalFormat(_spec.format));
}

atcg::ref_ptr<Texture2D> Texture2D::create(const TextureSpecification& spec)
{
    return create(nullptr, spec);
}

atcg::ref_ptr<Texture2D> Texture2D::create(const void* data, const TextureSpecification& spec)
{
    atcg::ref_ptr<Texture2D> result = atcg::make_ref<Texture2D>();
    result->_spec                   = spec;

    glGenTextures(1, &(result->_ID));
    glBindTexture(GL_TEXTURE_2D, result->_ID);

    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 detail::to2GLinternalFormat(spec.format),
                 spec.width,
                 spec.height,
                 0,
                 detail::toGLformat(spec.format),
                 detail::toGLtype(spec.format),
                 (void*)data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, detail::toGLWrapMode(spec.sampler.wrap_mode));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, detail::toGLWrapMode(spec.sampler.wrap_mode));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, detail::toGLFilterMode(spec.sampler.filter_mode));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, detail::toGLFilterMode(spec.sampler.filter_mode));

    return result;
}

Texture2D::~Texture2D()
{
    glDeleteTextures(1, &_ID);
}

void Texture2D::setData(const void* data)
{
    glBindTexture(GL_TEXTURE_2D, _ID);

    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 detail::to2GLinternalFormat(_spec.format),
                 _spec.width,
                 _spec.height,
                 0,
                 detail::toGLformat(_spec.format),
                 detail::toGLtype(_spec.format),
                 (void*)data);
}

void Texture2D::use(const uint32_t& slot) const
{
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(GL_TEXTURE_2D, _ID);
}

atcg::ref_ptr<Texture3D> Texture3D::create(const TextureSpecification& spec)
{
    return create(nullptr, spec);
}

atcg::ref_ptr<Texture3D> Texture3D::create(const void* data, const TextureSpecification& spec)
{
    atcg::ref_ptr<Texture3D> result = atcg::make_ref<Texture3D>();
    result->_spec                   = spec;

    glGenTextures(1, &(result->_ID));
    glBindTexture(GL_TEXTURE_3D, result->_ID);

    glTexImage3D(GL_TEXTURE_3D,
                 0,
                 detail::to2GLinternalFormat(spec.format),
                 spec.width,
                 spec.height,
                 spec.depth,
                 0,
                 detail::toGLformat(spec.format),
                 detail::toGLtype(spec.format),
                 (void*)data);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, detail::toGLWrapMode(spec.sampler.wrap_mode));
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, detail::toGLWrapMode(spec.sampler.wrap_mode));
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, detail::toGLWrapMode(spec.sampler.wrap_mode));
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, detail::toGLFilterMode(spec.sampler.filter_mode));
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, detail::toGLFilterMode(spec.sampler.filter_mode));

    return result;
}

Texture3D::~Texture3D()
{
    glDeleteTextures(1, &_ID);
}

void Texture3D::setData(const void* data)
{
    glBindTexture(GL_TEXTURE_3D, _ID);

    glTexImage3D(GL_TEXTURE_3D,
                 0,
                 detail::to2GLinternalFormat(_spec.format),
                 _spec.width,
                 _spec.height,
                 _spec.depth,
                 0,
                 detail::toGLformat(_spec.format),
                 detail::toGLtype(_spec.format),
                 (void*)data);
}

void Texture3D::use(const uint32_t& slot) const
{
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(GL_TEXTURE_3D, _ID);
}

}    // namespace atcg