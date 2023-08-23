#include <Renderer/Texture.h>

#include <glad/glad.h>

namespace atcg
{

void Texture::use(const uint32_t& slot) const
{
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(_target, _ID);
}

void Texture::useForCompute(const uint32_t& slot) const
{
    glBindImageTexture(slot, _ID, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
}

atcg::ref_ptr<Texture2D> Texture2D::createColorTexture(uint32_t width, uint32_t height)
{
    return createColorTexture(nullptr, width, height);
}

atcg::ref_ptr<Texture2D> Texture2D::createColorTexture(const glm::u8vec4* data, uint32_t width, uint32_t height)
{
    atcg::ref_ptr<Texture2D> result = atcg::make_ref<Texture2D>();

    result->_width  = width;
    result->_height = height;
    result->_depth  = 1;
    result->_target = GL_TEXTURE_2D;

    glGenTextures(1, &(result->_ID));
    glBindTexture(GL_TEXTURE_2D, result->_ID);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*)data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return result;
}

atcg::ref_ptr<Texture2D> Texture2D::createDepthTexture(uint32_t width, uint32_t height)
{
    atcg::ref_ptr<Texture2D> result = atcg::make_ref<Texture2D>();

    result->_width  = width;
    result->_height = height;
    result->_depth  = 1;
    result->_target = GL_TEXTURE_2D;

    glGenTextures(1, &(result->_ID));
    glBindTexture(GL_TEXTURE_2D, result->_ID);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return result;
}

atcg::ref_ptr<Texture2D> Texture2D::createIntTexture(uint32_t width, uint32_t height)
{
    atcg::ref_ptr<Texture2D> result = atcg::make_ref<Texture2D>();

    result->_width  = width;
    result->_height = height;
    result->_depth  = 1;
    result->_target = GL_TEXTURE_2D;

    glGenTextures(1, &(result->_ID));
    glBindTexture(GL_TEXTURE_2D, result->_ID);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, width, height, 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return result;
}

atcg::ref_ptr<Texture2D> Texture2D::createFloatTexture(uint32_t width, uint32_t height)
{
    atcg::ref_ptr<Texture2D> result = atcg::make_ref<Texture2D>();

    result->_width  = width;
    result->_height = height;
    result->_depth  = 1;
    result->_target = GL_TEXTURE_2D;

    glGenTextures(1, &(result->_ID));
    glBindTexture(GL_TEXTURE_2D, result->_ID);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return result;
}

Texture2D::~Texture2D()
{
    glDeleteTextures(1, &_ID);
}

void Texture2D::setData(const void* data)
{
    glBindTexture(GL_TEXTURE_2D, _ID);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, _width, _height, 0, GL_RED, GL_FLOAT, data);
}

atcg::ref_ptr<Texture3D> Texture3D::createColorTexture(uint32_t width, uint32_t height, uint32_t depth)
{
    atcg::ref_ptr<Texture3D> result = atcg::make_ref<Texture3D>();

    result->_width  = width;
    result->_height = height;
    result->_depth  = depth;
    result->_target = GL_TEXTURE_3D;

    glGenTextures(1, &(result->_ID));
    glBindTexture(GL_TEXTURE_3D, result->_ID);

    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, width, height, depth, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return result;
}

atcg::ref_ptr<Texture3D> Texture3D::createFloatTexture(uint32_t width, uint32_t height, uint32_t depth)
{
    atcg::ref_ptr<Texture3D> result = atcg::make_ref<Texture3D>();

    result->_width  = width;
    result->_height = height;
    result->_depth  = depth;
    result->_target = GL_TEXTURE_3D;

    glGenTextures(1, &(result->_ID));
    glBindTexture(GL_TEXTURE_3D, result->_ID);

    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, width, height, depth, 0, GL_RED, GL_FLOAT, nullptr);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return result;
}

Texture3D::~Texture3D()
{
    glDeleteTextures(1, &_ID);
}

void Texture3D::setData(const void* data)
{
    glBindTexture(GL_TEXTURE_3D, _ID);

    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, _width, _height, _depth, 0, GL_RED, GL_FLOAT, data);
}

}    // namespace atcg