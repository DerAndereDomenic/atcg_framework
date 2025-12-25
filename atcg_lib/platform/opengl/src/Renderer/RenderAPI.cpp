#include <Renderer/RenderAPI.h>

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
    target ? target->use() : Framebuffer::useDefault();
}

void RenderAPI::endRenderPass()
{
    // Nothing to do for OpenGL
}

void RenderAPI::setPipeline(const GraphicsPipeline& pipeline)
{
    pipeline.shader->use();

    if(_current_pipeline == pipeline)
    {
        return;
    }

    _current_pipeline = pipeline;

    if(_current_pipeline.render_state.culling_enabled)
    {
        glEnable(GL_CULL_FACE);
    }
    else
    {
        glDisable(GL_CULL_FACE);
    }

    switch(_current_pipeline.render_state.cull_mode)
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

    if(_current_pipeline.render_state.depth_state.depth_testing_enabled)
    {
        glEnable(GL_DEPTH_TEST);
    }
    else
    {
        glDisable(GL_DEPTH_TEST);
    }

    if(_current_pipeline.render_state.depth_state.depth_write_enabled)
    {
        glDepthMask(GL_TRUE);
        glDepthFunc(detail::toGLDepthFunction(_current_pipeline.render_state.depth_state.depth_function));
    }
    else
    {
        glDepthMask(GL_FALSE);
    }

    if(_current_pipeline.render_state.blend_state.blend_enabled)
    {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }
    else
    {
        glDisable(GL_BLEND);
    }

    glPointSize(_current_pipeline.render_state.point_size);
    glLineWidth(_current_pipeline.render_state.line_size);
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
    vao->use();
}

void RenderAPI::bindTexture(uint32_t slot, const atcg::ref_ptr<Texture>& texture)
{
    texture->use(slot);
}

void RenderAPI::draw(uint32_t vertexCount)
{
    glDrawArrays(detail::toGLPrimitive(_current_pipeline.primitive_type), 0, static_cast<GLsizei>(vertexCount));
}

void RenderAPI::drawIndexed(uint32_t indexCount)
{
    glDrawElements(detail::toGLPrimitive(_current_pipeline.primitive_type),
                   static_cast<GLsizei>(indexCount),
                   GL_UNSIGNED_INT,
                   (void*)0);
}

void RenderAPI::drawInstanced(uint32_t vertexCount, uint32_t nInstances)
{
    glDrawArraysInstanced(detail::toGLPrimitive(_current_pipeline.primitive_type),
                          0,
                          static_cast<GLsizei>(vertexCount),
                          nInstances);
}

void RenderAPI::drawIndexedInstanced(uint32_t indexCount, uint32_t nInstances)
{
    glDrawElementsInstanced(detail::toGLPrimitive(_current_pipeline.primitive_type),
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
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
}    // namespace atcg