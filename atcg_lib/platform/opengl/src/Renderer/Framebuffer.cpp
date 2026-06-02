#include <Renderer/Framebuffer.h>

#include <Core/Assert.h>
#include <Renderer/ContextManager.h>

#include <glad/glad.h>

namespace atcg
{

Framebuffer::Framebuffer() : _width(0), _height(0)
{
    glGenFramebuffers(1, &_ID);
}

Framebuffer::Framebuffer(uint32_t width, uint32_t height) : _width(width), _height(height)
{
    glGenFramebuffers(1, &_ID);
}

atcg::ref_ptr<Framebuffer> Framebuffer::create(const FramebufferSpecification& spec)
{
    auto result = atcg::make_ref<Framebuffer>(spec.width, spec.height);

    for(auto attachement: spec.attachements)
    {
        TextureSpecification texture_specs = attachement.spec;
        texture_specs.width                = spec.width;
        texture_specs.height               = spec.height;
        texture_specs.depth                = spec.depth;
        texture_specs.num_samples          = spec.num_samples;

        atcg::ref_ptr<Texture> texture = Texture::create(attachement.type, texture_specs);

        if(attachement.spec.format == TextureFormat::DEPTH)
        {
            result->attachDepth(texture);
        }
        else
        {
            result->attachTexture(texture);
        }
    }

    result->complete();
    return result;
}

Framebuffer::~Framebuffer()
{
    glDeleteFramebuffers(1, &_ID);
    _color_attachements.clear();
}

void Framebuffer::bind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, _ID);
    auto context = ContextManager::getCurrentContext();
    context->setCurrentFBO(shared_from_this());
}

bool Framebuffer::complete()
{
    bind();

    std::vector<GLenum> buffers(_color_attachements.size());
    for(uint32_t i = 0; i < _color_attachements.size(); ++i)
    {
        buffers[i] = (GL_COLOR_ATTACHMENT0 + i);
    }
    glDrawBuffers(buffers.size(), buffers.data());

    uint32_t error = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    bool complete  = error == GL_FRAMEBUFFER_COMPLETE;
    if(!complete)
    {
        ATCG_ERROR("ERROR: Framebuffer not complete! Code: {}", error);
    }
    bindDefault();
    return complete;
}

void Framebuffer::attachColor()
{
    bind();

    TextureSpecification spec;
    spec.width                       = _width;
    spec.height                      = _height;
    atcg::ref_ptr<Texture2D> texture = Texture2D::create(spec);
    attachTexture(texture);
}

void Framebuffer::attachColorMultiSample(uint32_t num_samples)
{
    bind();

    TextureSpecification spec;
    spec.width                                  = _width;
    spec.height                                 = _height;
    spec.num_samples                            = num_samples;
    atcg::ref_ptr<Texture2DMultiSample> texture = Texture2DMultiSample::create(spec);
    attachTexture(texture);
}

void Framebuffer::attachTexture(const atcg::ref_ptr<Texture>& texture)
{
    bind();
    glFramebufferTexture(GL_FRAMEBUFFER,
                         GL_COLOR_ATTACHMENT0 + static_cast<GLenum>(_color_attachements.size()),
                         texture->getID(),
                         0);
    _color_attachements.push_back(texture);
    bindDefault();
}

void Framebuffer::attachCubeFace(const atcg::ref_ptr<TextureCube>& cube_map, uint32_t face_index, uint32_t mip_level)
{
    glFramebufferTexture2D(GL_FRAMEBUFFER,
                           GL_COLOR_ATTACHMENT0 + static_cast<GLenum>(_color_attachements.size()),
                           GL_TEXTURE_CUBE_MAP_POSITIVE_X + face_index,
                           cube_map->getID(),
                           mip_level);
    _color_attachements.push_back(cube_map);
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

void Framebuffer::attachDepthMultiSample(uint32_t num_samples)
{
    TextureSpecification spec;
    spec.width         = _width;
    spec.height        = _height;
    spec.format        = TextureFormat::DEPTH;
    spec.num_samples   = num_samples;
    _depth_attachement = Texture2DMultiSample::create(spec);
    attachDepth(_depth_attachement);
}

void Framebuffer::attachDepth(const atcg::ref_ptr<Texture>& depth_map)
{
    bind();
    _depth_attachement = depth_map;
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, _depth_attachement->getID(), 0);
    bindDefault();
}

void Framebuffer::detachColor()
{
    if(_color_attachements.size() == 0)
    {
        ATCG_WARN("No color attachements to detach");
        return;
    }

    uint32_t last_index = static_cast<uint32_t>(_color_attachements.size() - 1);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + last_index, 0, 0);
    _color_attachements.pop_back();
}

void Framebuffer::detachDepth()
{
    if(!_depth_attachement)
    {
        ATCG_WARN("No depth attachement to detach");
        return;
    }

    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, 0, 0);
    _depth_attachement = nullptr;
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

    for(int i = 0; i < _color_attachements.size(); ++i)
    {
        glReadBuffer(GL_COLOR_ATTACHMENT0 + i);
        glDrawBuffer(GL_COLOR_ATTACHMENT0 + i);
        glBlitFramebuffer(0, 0, _width, _height, 0, 0, _width, _height, flags, GL_NEAREST);
    }

    glReadBuffer(GL_COLOR_ATTACHMENT0);
    std::vector<GLenum> buffers(_color_attachements.size());
    for(uint32_t i = 0; i < _color_attachements.size(); ++i)
    {
        buffers[i] = (GL_COLOR_ATTACHMENT0 + i);
    }
    glDrawBuffers(buffers.size(), buffers.data());

    // Restore old binding status
    atcg::ref_ptr<Framebuffer> current_fbo = currentFramebuffer();
    glBindFramebuffer(GL_READ_FRAMEBUFFER, current_fbo ? current_fbo->getID() : 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, current_fbo ? current_fbo->getID() : 0);
}

void Framebuffer::bindDefault()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    auto context = ContextManager::getCurrentContext();
    context->setCurrentFBO(nullptr);
}

atcg::ref_ptr<Framebuffer> Framebuffer::currentFramebuffer()
{
    auto context = ContextManager::getCurrentContext();
    return context->getCurrentFBO();
}
}    // namespace atcg