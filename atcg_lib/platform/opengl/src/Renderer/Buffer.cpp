#include <Renderer/Buffer.h>

#include <glad/glad.h>

namespace atcg
{
VertexBuffer::VertexBuffer(size_t size) : _size(size)
{
    glGenBuffers(1, &_ID);
    glBindBuffer(GL_ARRAY_BUFFER, _ID);
    glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
}

VertexBuffer::VertexBuffer(const void* data, size_t size) : _size(size)
{
    glGenBuffers(1, &_ID);
    glBindBuffer(GL_ARRAY_BUFFER, _ID);
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
}

VertexBuffer::~VertexBuffer()
{
    glDeleteBuffers(1, &_ID);
}

void VertexBuffer::use() const
{
    glBindBuffer(GL_ARRAY_BUFFER, _ID);
}

void VertexBuffer::setData(const void* data, size_t size)
{
    _size = size;
    glBindBuffer(GL_ARRAY_BUFFER, _ID);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size, data);
}

IndexBuffer::IndexBuffer(const uint32_t* indices, size_t count) : _count(count)
{
    glGenBuffers(1, &_ID);
    glBindBuffer(GL_ARRAY_BUFFER, _ID);
    glBufferData(GL_ARRAY_BUFFER, count * sizeof(uint32_t), indices, GL_STATIC_DRAW);
}

IndexBuffer::IndexBuffer(size_t count) : _count(count)
{
    glGenBuffers(1, &_ID);
    glBindBuffer(GL_ARRAY_BUFFER, _ID);
    glBufferData(GL_ARRAY_BUFFER, count * sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);
}

IndexBuffer::~IndexBuffer()
{
    glDeleteBuffers(1, &_ID);
}

void IndexBuffer::use() const
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ID);
}

void IndexBuffer::setData(const uint32_t* data, size_t count)
{
    _count = count;
    glBindBuffer(GL_ARRAY_BUFFER, _ID);
    glBufferSubData(GL_ARRAY_BUFFER, 0, count * sizeof(uint32_t), data);
}

}    // namespace atcg