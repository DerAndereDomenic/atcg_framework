#include <Renderer/Texture.h>

#include <glad/glad.h>

namespace atcg
{
    std::shared_ptr<Texture2D> Texture2D::createColorTexture(uint32_t width, uint32_t height)
    {
        std::shared_ptr<Texture2D> result = std::make_shared<Texture2D>();

        result->_width = width;
        result->_height = height;

        glGenTextures(1, &(result->_ID));
        glBindTexture(GL_TEXTURE_2D, result->_ID);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        return result;
    }

    std::shared_ptr<Texture2D> Texture2D::createDepthTexture(uint32_t width, uint32_t height)
    {
        std::shared_ptr<Texture2D> result = std::make_shared<Texture2D>();

        result->_width = width;
        result->_height = height;

        glGenTextures(1, &(result->_ID));
        glBindTexture(GL_TEXTURE_2D, result->_ID);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH, width, height, 0, GL_DEPTH, GL_FLOAT, nullptr);
        glGenerateMipmap(GL_TEXTURE_2D);

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
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}