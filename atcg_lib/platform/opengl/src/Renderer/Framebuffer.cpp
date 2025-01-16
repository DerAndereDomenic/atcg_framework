#include <Renderer/Framebuffer.h>

#include <Core/Assert.h>

#include <glad/glad.h>

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
    for(uint32_t i = 0; i < _color_attachements.size(); ++i)
    {
        buffers.push_back(GL_COLOR_ATTACHMENT0 + i);
    }
    glDrawBuffers(buffers.size(), buffers.data());

    uint32_t error = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    bool complete  = error == GL_FRAMEBUFFER_COMPLETE;
    if(!complete)
    {
        ATCG_ERROR("ERROR: Framebuffer not complete! Code: {}", error);
    }
    useDefault();
    return complete;
}

void Framebuffer::attachColor()
{
    use();

    TextureSpecification spec;
    spec.width                       = _width;
    spec.height                      = _height;
    atcg::ref_ptr<Texture2D> texture = Texture2D::create(spec);
    attachTexture(texture);
}

void Framebuffer::attachTexture(const atcg::ref_ptr<Texture>& texture)
{
    use();
    glFramebufferTexture(GL_FRAMEBUFFER,
                         GL_COLOR_ATTACHMENT0 + static_cast<GLenum>(_color_attachements.size()),
                         texture->getID(),
                         0);
    _color_attachements.push_back(texture);
    useDefault();
}

void Framebuffer::attachDepth()
{
    TextureSpecification spec;
    spec.width         = _width;
    spec.height        = _height;
    spec.format        = TextureFormat::DEPTH;
    _depth_attachement = Texture2D::create(spec);
    attachDepth(_depth_attachement);
}

void Framebuffer::attachDepth(const atcg::ref_ptr<Texture>& depth_map)
{
    use();
    _depth_attachement = depth_map;
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, _depth_attachement->getID(), 0);
    useDefault();
}

void Framebuffer::blit(const atcg::ref_ptr<Framebuffer>& source, bool color, bool depth)
{
    ATCG_ASSERT((source->width() == _width) && (source->height() == _height),
                "Can only blit Framebuffers of same size");

    GLbitfield flags = 0;
    if(color) flags |= GL_COLOR_BUFFER_BIT;
    if(depth) flags |= GL_DEPTH_BUFFER_BIT;

    glBindFramebuffer(GL_READ_FRAMEBUFFER, source->getID());
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, _ID);
    glBlitFramebuffer(0, 0, _width, _height, 0, 0, _width, _height, flags, GL_NEAREST);
}

void Framebuffer::bindByID(uint32_t fbo_id)
{
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_id);
    s_current_fbo = fbo_id;
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