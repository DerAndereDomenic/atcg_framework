#include <Renderer/VertexArray.h>

#include <glad/glad.h>

#include <iostream>

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

void VertexArray::addVertexBuffer(const atcg::ref_ptr<VertexBuffer>& vbo)
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
                std::cerr << "Unknown ShaderDataType\n";
        }
    }

    _vertex_buffers.push_back(vbo);
    _vertex_buffer_index.push_back({vertex_buffer_index, 0});
}

void VertexArray::setIndexBuffer(const atcg::ref_ptr<IndexBuffer>& ibo)
{
    glBindVertexArray(_ID);
    ibo->use();
    _ibo = ibo;
}

void VertexArray::addInstanceBuffer(const atcg::ref_ptr<VertexBuffer>& vbo)
{
    addVertexBuffer(vbo);
    markInstance(static_cast<uint32_t>(_vertex_buffers.size()) - 1, 1);
}

void VertexArray::markInstance(uint32_t buffer_idx, uint32_t divisor)
{
    uint32_t curr_divisor = _vertex_buffer_index[buffer_idx].divisor;
    if(curr_divisor == divisor) { return; }

    this->use();
    _vertex_buffer_index[buffer_idx].divisor = divisor;
    const auto& layout                       = _vertex_buffers[buffer_idx]->getLayout();
    for(uint32_t i = 0; i < layout.getElements().size(); ++i)
    {
        glVertexAttribDivisor(_vertex_buffer_index[buffer_idx].vertex_buffer_index - 1 - i, divisor);
    }
}
}    // namespace atcg