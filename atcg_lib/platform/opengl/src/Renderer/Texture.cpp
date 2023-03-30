#include <Renderer/Texture.h>

#include <glad/glad.h>

namespace atcg
{
atcg::ref_ptr<Texture2D> Texture2D::createColorTexture(uint32_t width, uint32_t height)
{
    atcg::ref_ptr<Texture2D> result = atcg::make_ref<Texture2D>();

    result->_width  = width;
    result->_height = height;

    glGenTextures(1, &(result->_ID));
    glBindTexture(GL_TEXTURE_2D, result->_ID);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return result;
}

atcg::ref_ptr<Texture2D> Texture2D::createDepthTexture(uint32_t width, uint32_t height)
{
    atcg::ref_ptr<Texture2D> result = atcg::make_ref<Texture2D>();

    result->_width  = width;
    result->_height = height;

    glGenTextures(1, &(result->_ID));
    glBindTexture(GL_TEXTURE_2D, result->_ID);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

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

void Texture2D::use(const uint32_t& slot) const
{
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(GL_TEXTURE_2D, _ID);
}
}    // namespace atcg