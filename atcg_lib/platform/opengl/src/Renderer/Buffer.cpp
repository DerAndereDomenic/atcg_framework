#include <Renderer/Buffer.h>

#include <glad/glad.h>

#include <Core/CUDA.h>
#include <cuda_gl_interop.h>

namespace atcg
{

class VertexBuffer::Impl
{
public:
    Impl() = default;

    Impl(uint32_t ID);

    ~Impl();

    void initResource(uint32_t ID);

    void mapResource();

    void unmapResource();

    bool mapped = false;

#ifdef ATCG_CUDA_BACKEND
    cudaGraphicsResource* resource = nullptr;
#endif
};

VertexBuffer::Impl::Impl(uint32_t ID)
{
    initResource(ID);
}

VertexBuffer::Impl::~Impl() {}

void VertexBuffer::Impl::initResource(uint32_t ID)
{
#ifdef ATCG_CUDA_BACKEND
    atcg::cudaSafeCall(cudaGraphicsGLRegisterBuffer(&resource, ID, cudaGraphicsRegisterFlagsNone));
#endif
}

void VertexBuffer::Impl::mapResource()
{
#ifdef ATCG_CUDA_BACKEND
    atcg::cudaSafeCall(cudaGraphicsMapResources(1, &resource));
    mapped = true;
#endif
}

void VertexBuffer::Impl::unmapResource()
{
#ifdef ATCG_CUDA_BACKEND
    if(mapped)
    {
        atcg::cudaSafeCall(cudaGraphicsUnmapResources(1, &resource));
        mapped = false;
    }
#endif
}

VertexBuffer::VertexBuffer(size_t size) : _size(size)
{
    glGenBuffers(1, &_ID);
    glBindBuffer(GL_ARRAY_BUFFER, _ID);
    glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);

    impl = atcg::make_scope<Impl>(_ID);
}

VertexBuffer::VertexBuffer(const void* data, size_t size) : _size(size)
{
    glGenBuffers(1, &_ID);
    glBindBuffer(GL_ARRAY_BUFFER, _ID);
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);

    impl = atcg::make_scope<Impl>(_ID);
}

VertexBuffer::~VertexBuffer()
{
    glDeleteBuffers(1, &_ID);
}

void VertexBuffer::use() const
{
    impl->unmapResource();
    glBindBuffer(GL_ARRAY_BUFFER, _ID);
}

void VertexBuffer::bindStorage(uint32_t slot) const
{
    impl->unmapResource();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, slot, _ID);
}

void VertexBuffer::setData(const void* data, size_t size)
{
    impl->unmapResource();
    _size = size;
    glBindBuffer(GL_ARRAY_BUFFER, _ID);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size, data);
}

void* VertexBuffer::getData() const
{
    impl->mapResource();
    void* dev_ptr = nullptr;
    std::size_t size;
#ifdef ATCG_CUDA_BACKEND
    atcg::cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&dev_ptr, &size, impl->resource));
#endif
    return dev_ptr;
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