#include <Renderer/VertexArray.h>

#include <glad/glad.h>

namespace atcg
{
static GLenum shaderDataTypeToOpenGLBaseType(ShaderDataType type)
{
    switch(type)
    {
        case ShaderDataType::Float:
            return GL_FLOAT;
        case ShaderDataType::Float2:
            return GL_FLOAT;
        case ShaderDataType::Float3:
            return GL_FLOAT;
        case ShaderDataType::Float4:
            return GL_FLOAT;
        case ShaderDataType::Mat3:
            return GL_FLOAT;
        case ShaderDataType::Mat4:
            return GL_FLOAT;
        case ShaderDataType::Int:
            return GL_INT;
        case ShaderDataType::Int2:
            return GL_INT;
        case ShaderDataType::Int3:
            return GL_INT;
        case ShaderDataType::Int4:
            return GL_INT;
        case ShaderDataType::Bool:
            return GL_BOOL;
    }

    return 0;
}

VertexArray::VertexArray()
{
    glCreateVertexArrays(1, &_ID);
}

VertexArray::~VertexArray()
{
    glDeleteVertexArrays(1, &_ID);
}

void VertexArray::use() const
{
    glBindVertexArray(_ID);
}

void VertexArray::pushVertexBuffer(const atcg::ref_ptr<VertexBuffer>& vbo)
{
    glBindVertexArray(_ID);
    vbo->use();

    uint32_t vertex_buffer_index = _vertex_buffer_index.empty() ? 0 : _vertex_buffer_index.back().vertex_buffer_index;

    const auto& layout = vbo->getLayout();
    for(const auto& element: layout)
    {
        switch(element.type)
        {
            case ShaderDataType::Float:
            case ShaderDataType::Float2:
            case ShaderDataType::Float3:
            case ShaderDataType::Float4:
            {
                glEnableVertexAttribArray(vertex_buffer_index);
                glVertexAttribPointer(vertex_buffer_index,
                                      element.getComponentCount(),
                                      shaderDataTypeToOpenGLBaseType(element.type),
                                      element.normalized ? GL_TRUE : GL_FALSE,
                                      layout.getStride(),
                                      (const void*)element.offset);
                vertex_buffer_index++;
                break;
            }
            case ShaderDataType::Int:
            case ShaderDataType::Int2:
            case ShaderDataType::Int3:
            case ShaderDataType::Int4:
            case ShaderDataType::Bool:
            {
                glEnableVertexAttribArray(vertex_buffer_index);
                glVertexAttribIPointer(vertex_buffer_index,
                                       element.getComponentCount(),
                                       shaderDataTypeToOpenGLBaseType(element.type),
                                       layout.getStride(),
                                       (const void*)element.offset);
                vertex_buffer_index++;
                break;
            }
            case ShaderDataType::Mat3:
            case ShaderDataType::Mat4:
            {
                uint8_t count = element.getComponentCount();
                for(uint8_t i = 0; i < count; i++)
                {
                    glEnableVertexAttribArray(vertex_buffer_index);
                    glVertexAttribPointer(vertex_buffer_index,
                                          count,
                                          shaderDataTypeToOpenGLBaseType(element.type),
                                          element.normalized ? GL_TRUE : GL_FALSE,
                                          layout.getStride(),
                                          (const void*)(element.offset + sizeof(float) * count * i));
                    glVertexAttribDivisor(vertex_buffer_index, 1);
                    vertex_buffer_index++;
                }
                break;
            }
            default:
                ATCG_WARN("Unknown ShaderDataType");
        }
    }

    _vertex_buffers.push_back(vbo);
    _vertex_buffer_index.push_back({vertex_buffer_index, 0});
}

atcg::ref_ptr<VertexBuffer> VertexArray::popVertexBuffer()
{
    glBindVertexArray(_ID);
    atcg::ref_ptr<VertexBuffer> last_buffer = _vertex_buffers.back();
    VertexBufferIndexing indexing           = _vertex_buffer_index.back();

    _vertex_buffers.pop_back();
    _vertex_buffer_index.pop_back();

    for(uint32_t i = _vertex_buffer_index.back().vertex_buffer_index; i < indexing.vertex_buffer_index; ++i)
    {
        glDisableVertexAttribArray(i);
    }

    return last_buffer;
}

const atcg::ref_ptr<VertexBuffer>& VertexArray::peekVertexBuffer() const
{
    return _vertex_buffers.back();
}

void VertexArray::setIndexBuffer(const atcg::ref_ptr<IndexBuffer>& ibo)
{
    glBindVertexArray(_ID);
    ibo->use();
    _ibo = ibo;
}

void VertexArray::pushInstanceBuffer(const atcg::ref_ptr<VertexBuffer>& vbo)
{
    pushVertexBuffer(vbo);
    markInstance(1);
}

// TODO: FIXME: Matrix is not handled correctly
void VertexArray::markInstance(uint32_t divisor)
{
    uint32_t curr_divisor = _vertex_buffer_index.back().divisor;
    if(curr_divisor == divisor)
    {
        return;
    }

    this->use();
    _vertex_buffer_index.back().divisor = divisor;
    const auto& layout                  = _vertex_buffers.back()->getLayout();
    for(uint32_t i = 0; i < layout.getElements().size(); ++i)
    {
        glVertexAttribDivisor(_vertex_buffer_index.back().vertex_buffer_index - 1 - i, divisor);
    }
}
}    // namespace atcg