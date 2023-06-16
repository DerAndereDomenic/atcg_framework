#include <Renderer/Framebuffer.h>

#include <glad/glad.h>
#include <iostream>

namespace atcg
{

uint32_t Framebuffer::s_current_fbo = 0;

Framebuffer::Framebuffer(uint32_t width, uint32_t height) : _width(width), _height(height)
{
    glGenFramebuffers(1, &_ID);
}

Framebuffer::~Framebuffer()
{
    glDeleteFramebuffers(1, &_ID);
    _color_attachements.clear();
}

void Framebuffer::use() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, _ID);
    s_current_fbo = _ID;
}

bool Framebuffer::complete() const
{
    use();

    std::vector<GLenum> buffers;
    for(uint32_t i = 0; i < _color_attachements.size(); ++i) { buffers.push_back(GL_COLOR_ATTACHMENT0 + i); }
    glDrawBuffers(buffers.size(), buffers.data());

    uint32_t error = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    bool complete  = error == GL_FRAMEBUFFER_COMPLETE;
    if(!complete) { ATCG_ERROR("ERROR: Framebuffer not complete!"); }
    useDefault();
    return complete;
}

void Framebuffer::attachColor()
{
    use();
    atcg::ref_ptr<Texture2D> texture = Texture2D::createColorTexture(_width, _height);
    glFramebufferTexture2D(GL_FRAMEBUFFER,
                           GL_COLOR_ATTACHMENT0 + static_cast<GLenum>(_color_attachements.size()),
                           GL_TEXTURE_2D,
                           texture->getID(),
                           0);
    _color_attachements.push_back(texture);
    useDefault();
}

void Framebuffer::attachTexture(const atcg::ref_ptr<Texture>& texture)
{
    use();
    glFramebufferTexture2D(GL_FRAMEBUFFER,
                           GL_COLOR_ATTACHMENT0 + static_cast<GLenum>(_color_attachements.size()),
                           GL_TEXTURE_2D,
                           texture->getID(),
                           0);
    _color_attachements.push_back(texture);
    useDefault();
}

void Framebuffer::attachDepth()
{
    use();
    _depth_attachement = Texture2D::createDepthTexture(_width, _height);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, _depth_attachement->getID(), 0);
    useDefault();
}

void Framebuffer::useDefault()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    s_current_fbo = 0;
}

uint32_t Framebuffer::currentFramebuffer()
{
    return s_current_fbo;
}
}    // namespace atcg