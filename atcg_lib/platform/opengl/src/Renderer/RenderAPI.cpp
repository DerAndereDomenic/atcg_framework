#include <Renderer/RenderAPI.h>
#include <Core/Assert.h>

#include <glad/glad.h>

namespace atcg
{
namespace detail
{
static GLenum toGLPrimitive(PrimitiveTopology topo)
{
    switch(topo)
    {
        case PrimitiveTopology::ATCG_TRIANGLES:
            return GL_TRIANGLES;
        case PrimitiveTopology::ATCG_POINTS:
            return GL_POINTS;
        case PrimitiveTopology::ATCG_LINES:
            return GL_LINES;
    }
    return GL_TRIANGLES;
}

static GLenum toGLDepthFunction(DepthFunction func)
{
    switch(func)
    {
        case DepthFunction::ATCG_LESS:
            return GL_LESS;
        case DepthFunction::ATCG_LEQUAL:
            return GL_LEQUAL;
        case DepthFunction::ATCG_EQUAL:
            return GL_EQUAL;
        case DepthFunction::ATCG_GREATER:
            return GL_GREATER;
        case DepthFunction::ATCG_GEQUAL:
            return GL_GEQUAL;
        case DepthFunction::ATCG_ALWAYS:
            return GL_ALWAYS;
        case DepthFunction::ATCG_NEVER:
            return GL_NEVER;
        case DepthFunction::ATCG_NOTEQUAL:
            return GL_NOTEQUAL;
    }
    return GL_LESS;
}
}    // namespace detail

void RenderAPI::beginRenderPass(const atcg::ref_ptr<Framebuffer>& target)
{
    ATCG_ASSERT(!_started_render_pass, "Render pass already started");
    _started_render_pass = true;
    target ? target->bind() : Framebuffer::bindDefault();
    if(target)
    {
        setViewport(0, 0, target->width(), target->height());
    }
}

void RenderAPI::endRenderPass()
{
    ATCG_ASSERT(_started_render_pass, "Render pass not started");
    _started_render_pass = false;

    // Clear texture bindings
    for(const auto& binding: _bound_textures)
    {
        binding.texture->unbind(binding.slot);
    }
    _bound_textures.clear();

    Framebuffer::bindDefault();
}

void RenderAPI::bindPipeline(const GraphicsPipeline& pipeline)
{
    if(pipeline.shader) pipeline.shader->bind();

    if(_current_pipeline == pipeline)
    {
        return;
    }

    _current_pipeline = pipeline;

    if(_current_pipeline.rasterizer_state.culling_enabled)
    {
        glEnable(GL_CULL_FACE);
    }
    else
    {
        glDisable(GL_CULL_FACE);
    }

    switch(_current_pipeline.rasterizer_state.cull_mode)
    {
        case CullMode::ATCG_BACK_FACE_CULLING:
        {
            glCullFace(GL_BACK);
        }
        break;
        case CullMode::ATCG_FRONT_FACE_CULLING:
        {
            glCullFace(GL_FRONT);
        }
        break;
        case CullMode::ATCG_BOTH_FACE_CULLING:
        {
            glCullFace(GL_FRONT_AND_BACK);
        }
        break;
    }

    if(_current_pipeline.rasterizer_state.depth_state.depth_testing_enabled)
    {
        glEnable(GL_DEPTH_TEST);
    }
    else
    {
        glDisable(GL_DEPTH_TEST);
    }

    if(_current_pipeline.rasterizer_state.depth_state.depth_write_enabled)
    {
        glDepthMask(GL_TRUE);
    }
    else
    {
        glDepthMask(GL_FALSE);
    }
    glDepthFunc(detail::toGLDepthFunction(_current_pipeline.rasterizer_state.depth_state.depth_function));

    if(_current_pipeline.rasterizer_state.blend_state.blend_enabled)
    {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }
    else
    {
        glDisable(GL_BLEND);
    }

    glPointSize(_current_pipeline.rasterizer_state.point_size);
    glLineWidth(_current_pipeline.rasterizer_state.line_size);
}

void RenderAPI::setViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height)
{
    glViewport(x, y, width, height);
}

glm::ivec4 RenderAPI::getViewport() const
{
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    return glm::ivec4(viewport[0], viewport[1], viewport[2], viewport[3]);
}

void RenderAPI::bindVertexArray(const atcg::ref_ptr<VertexArray>& vao)
{
    vao->bind();
}

void RenderAPI::bindTexture(uint32_t slot, const atcg::ref_ptr<Texture>& texture)
{
    texture->bind(slot);
    _bound_textures.emplace_back(slot, texture);
}

void RenderAPI::bindStorageBuffer(uint32_t slot, const atcg::ref_ptr<VertexBuffer>& buffer)
{
    buffer->bindStorage(slot);
}

void RenderAPI::draw(uint32_t vertexCount)
{
    glDrawArrays(detail::toGLPrimitive(_current_pipeline.primitive_topology), 0, static_cast<GLsizei>(vertexCount));
}

void RenderAPI::drawIndexed(uint32_t indexCount)
{
    glDrawElements(detail::toGLPrimitive(_current_pipeline.primitive_topology),
                   static_cast<GLsizei>(indexCount),
                   GL_UNSIGNED_INT,
                   (void*)0);
}

void RenderAPI::drawInstanced(uint32_t vertexCount, uint32_t nInstances)
{
    glDrawArraysInstanced(detail::toGLPrimitive(_current_pipeline.primitive_topology),
                          0,
                          static_cast<GLsizei>(vertexCount),
                          nInstances);
}

void RenderAPI::drawIndexedInstanced(uint32_t indexCount, uint32_t nInstances)
{
    glDrawElementsInstanced(detail::toGLPrimitive(_current_pipeline.primitive_topology),
                            static_cast<GLsizei>(indexCount),
                            GL_UNSIGNED_INT,
                            (void*)0,
                            nInstances);
}

void RenderAPI::setClearColor(const glm::vec4& color)
{
    glClearColor(color.r, color.g, color.b, color.a);
}

void RenderAPI::setClearDepth(float depth)
{
    glClearDepth(depth);
}

void RenderAPI::clear()
{
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDepthMask(_current_pipeline.rasterizer_state.depth_state.depth_write_enabled ? GL_TRUE : GL_FALSE);
    _current_pipeline.rasterizer_state.depth_state.depth_testing_enabled ? glEnable(GL_DEPTH_TEST)
                                                                         : glDisable(GL_DEPTH_TEST);
}
}    // namespace atcg