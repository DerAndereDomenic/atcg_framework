#include <Renderer/Buffer.h>

#include <glad/glad.h>

#include <Core/CUDA.h>

#ifdef ATCG_CUDA_BACKEND
    #include <cuda_gl_interop.h>
#endif

namespace atcg
{

class VertexBuffer::Impl
{
public:
    Impl() = default;

    Impl(uint32_t ID);

    ~Impl();

    void initResource(uint32_t ID);

    void mapResourceDevice();
    void unmapResourceDevice();

    void mapResourceHost();
    void unmapResourceHost();

    bool mapped_device = false;
    bool mapped_host   = false;

    void* dev_ptr = nullptr;

#ifdef ATCG_CUDA_BACKEND
    cudaGraphicsResource* resource = nullptr;
#endif
    uint32_t ID;
};

VertexBuffer::Impl::Impl(uint32_t ID)
{
    initResource(ID);
}

VertexBuffer::Impl::~Impl() {}

void VertexBuffer::Impl::initResource(uint32_t ID)
{
#ifdef ATCG_CUDA_BACKEND
    CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&resource, ID, cudaGraphicsRegisterFlagsNone));
#endif
    this->ID = ID;
}

void VertexBuffer::Impl::mapResourceDevice()
{
#ifdef ATCG_CUDA_BACKEND
    CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &resource));
    mapped_device = true;
#else
    glBindBuffer(GL_ARRAY_BUFFER, ID);
    dev_ptr     = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
    mapped_host = true;
#endif
}

void VertexBuffer::Impl::unmapResourceDevice()
{
#ifdef ATCG_CUDA_BACKEND
    if(mapped_device) { CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &resource)); }
    mapped_device = false;
#else
    if(mapped_host)
    {
        glBindBuffer(GL_ARRAY_BUFFER, ID);
        glUnmapBuffer(GL_ARRAY_BUFFER);
    }
    mapped_host = false;
#endif
}

void VertexBuffer::Impl::mapResourceHost()
{
    glBindBuffer(GL_ARRAY_BUFFER, ID);
    dev_ptr     = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
    mapped_host = true;
}

void VertexBuffer::Impl::unmapResourceHost()
{
    if(mapped_host)
    {
        glBindBuffer(GL_ARRAY_BUFFER, ID);
        glUnmapBuffer(GL_ARRAY_BUFFER);
        mapped_host = false;
    }
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
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_DYNAMIC_DRAW);

    impl = atcg::make_scope<Impl>(_ID);
}

VertexBuffer::~VertexBuffer()
{
    impl->unmapResourceDevice();
    impl->unmapResourceHost();
#ifdef ATCG_CUDA_BACKEND
    CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(impl->resource));
#endif
    glDeleteBuffers(1, &_ID);
}

void VertexBuffer::use() const
{
    impl->unmapResourceDevice();
    impl->unmapResourceHost();
    glBindBuffer(GL_ARRAY_BUFFER, _ID);
}

void VertexBuffer::bindStorage(uint32_t slot) const
{
    impl->unmapResourceDevice();
    impl->unmapResourceHost();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, slot, _ID);
}

void VertexBuffer::setData(const void* data, size_t size)
{
    impl->unmapResourceDevice();
    impl->unmapResourceHost();
    _size = size;
    glBindBuffer(GL_ARRAY_BUFFER, _ID);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size, data);
}

void* VertexBuffer::getDevicePointer() const
{
#ifdef ATCG_CUDA_BACKEND
    impl->mapResourceDevice();
    std::size_t size;
    CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&impl->dev_ptr, &size, impl->resource));
#else
    impl->mapResourceHost();
#endif
    return impl->dev_ptr;
}

void* VertexBuffer::getHostPointer() const
{
    impl->mapResourceHost();
    return impl->dev_ptr;
}

void VertexBuffer::unmapPointers()
{
    impl->unmapResourceDevice();
    impl->unmapResourceHost();
}

class IndexBuffer::Impl
{
public:
    Impl() = default;

    Impl(uint32_t ID);

    ~Impl();

    void initResource(uint32_t ID);

    void mapResource();

    void unmapResource();

    bool mapped = false;

    void* dev_ptr = nullptr;

#ifdef ATCG_CUDA_BACKEND
    cudaGraphicsResource* resource = nullptr;
#endif
    uint32_t ID;
};

IndexBuffer::Impl::Impl(uint32_t ID)
{
    initResource(ID);
}

IndexBuffer::Impl::~Impl() {}

void IndexBuffer::Impl::initResource(uint32_t ID)
{
#ifdef ATCG_CUDA_BACKEND
    CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&resource, ID, cudaGraphicsRegisterFlagsNone));
#endif
    this->ID = ID;
}

void IndexBuffer::Impl::mapResource()
{
#ifdef ATCG_CUDA_BACKEND
    CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &resource));
#else
    glBindBuffer(GL_ARRAY_BUFFER, ID);
    dev_ptr = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
#endif
    mapped = true;
}

void IndexBuffer::Impl::unmapResource()
{
    if(mapped)
    {
#ifdef ATCG_CUDA_BACKEND
        CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &resource));
#else
        glBindBuffer(GL_ARRAY_BUFFER, ID);
        glUnmapBuffer(GL_ARRAY_BUFFER);
#endif
        mapped = false;
    }
}

IndexBuffer::IndexBuffer(const uint32_t* indices, size_t count) : _count(count)
{
    glGenBuffers(1, &_ID);
    glBindBuffer(GL_ARRAY_BUFFER, _ID);
    glBufferData(GL_ARRAY_BUFFER, count * sizeof(uint32_t), indices, GL_DYNAMIC_DRAW);

    impl = atcg::make_scope<Impl>(_ID);
}

IndexBuffer::IndexBuffer(size_t count) : _count(count)
{
    glGenBuffers(1, &_ID);
    glBindBuffer(GL_ARRAY_BUFFER, _ID);
    glBufferData(GL_ARRAY_BUFFER, count * sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);

    impl = atcg::make_scope<Impl>(_ID);
}

IndexBuffer::~IndexBuffer()
{
    impl->unmapResource();
#ifdef ATCG_CUDA_BACKEND
    CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(impl->resource));
#endif
    glDeleteBuffers(1, &_ID);
}

void IndexBuffer::use() const
{
    impl->unmapResource();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ID);
}

void IndexBuffer::setData(const uint32_t* data, size_t count)
{
    impl->unmapResource();
    _count = count;
    glBindBuffer(GL_ARRAY_BUFFER, _ID);
    glBufferSubData(GL_ARRAY_BUFFER, 0, count * sizeof(uint32_t), data);
}

uint32_t* IndexBuffer::getData() const
{
    impl->mapResource();
    std::size_t size;
#ifdef ATCG_CUDA_BACKEND
    CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&impl->dev_ptr, &size, impl->resource));
#endif
    return static_cast<uint32_t*>(impl->dev_ptr);
}


}    // namespace atcg