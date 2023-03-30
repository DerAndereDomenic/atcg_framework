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
                glEnableVertexAttribArray(_vertex_buffer_index);
                glVertexAttribPointer(_vertex_buffer_index,
                                      element.getComponentCount(),
                                      shaderDataTypeToOpenGLBaseType(element.type),
                                      element.normalized ? GL_TRUE : GL_FALSE,
                                      layout.getStride(),
                                      (const void*)element.offset);
                _vertex_buffer_index++;
                break;
            }
            case ShaderDataType::Int:
            case ShaderDataType::Int2:
            case ShaderDataType::Int3:
            case ShaderDataType::Int4:
            case ShaderDataType::Bool:
            {
                glEnableVertexAttribArray(_vertex_buffer_index);
                glVertexAttribIPointer(_vertex_buffer_index,
                                       element.getComponentCount(),
                                       shaderDataTypeToOpenGLBaseType(element.type),
                                       layout.getStride(),
                                       (const void*)element.offset);
                _vertex_buffer_index++;
                break;
            }
            case ShaderDataType::Mat3:
            case ShaderDataType::Mat4:
            {
                uint8_t count = element.getComponentCount();
                for(uint8_t i = 0; i < count; i++)
                {
                    glEnableVertexAttribArray(_vertex_buffer_index);
                    glVertexAttribPointer(_vertex_buffer_index,
                                          count,
                                          shaderDataTypeToOpenGLBaseType(element.type),
                                          element.normalized ? GL_TRUE : GL_FALSE,
                                          layout.getStride(),
                                          (const void*)(element.offset + sizeof(float) * count * i));
                    glVertexAttribDivisor(_vertex_buffer_index, 1);
                    _vertex_buffer_index++;
                }
                break;
            }
            default:
                std::cerr << "Unknown ShaderDataType\n";
        }
    }

    _vertex_buffers.push_back(vbo);
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
    const auto& layout = vbo->getLayout();
    for(uint32_t i = 0; i < layout.getElements().size(); ++i)
    {
        glVertexAttribDivisor(_vertex_buffer_index - 1 - i, 1);
    }
}
}    // namespace atcg