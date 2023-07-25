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
    void deinitResource();

    void mapResourceDevice();
    void unmapResourceDevice();

    void mapResourceHost();
    void unmapResourceHost();

    bool mapped_device  = false;
    bool mapped_host    = false;
    bool resource_ready = false;

    void* dev_ptr = nullptr;

    std::size_t size     = 0;
    std::size_t capacity = 0;

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
    this->ID       = ID;
    resource_ready = true;
}

void VertexBuffer::Impl::deinitResource()
{
#ifdef ATCG_CUDA_BACKEND
    CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(resource));
#endif
    this->ID       = 0;
    resource_ready = false;
}

void VertexBuffer::Impl::mapResourceDevice()
{
#ifdef ATCG_CUDA_BACKEND
    if(!mapped_device)
    {
        CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &resource));
        mapped_device = true;
    }
#else
    if(!mapped_host)
    {
        glBindBuffer(GL_ARRAY_BUFFER, ID);
        dev_ptr     = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
        mapped_host = true;
    }
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
    if(!mapped_host)
    {
        glBindBuffer(GL_ARRAY_BUFFER, ID);
        dev_ptr     = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
        mapped_host = true;
    }
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

VertexBuffer::VertexBuffer()
{
    glGenBuffers(1, &_ID);
    glBindBuffer(GL_ARRAY_BUFFER, _ID);

    impl           = atcg::make_scope<Impl>();
    impl->size     = 0;
    impl->capacity = 0;
}

VertexBuffer::VertexBuffer(size_t size)
{
    glGenBuffers(1, &_ID);
    glBindBuffer(GL_ARRAY_BUFFER, _ID);
    glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);

    impl           = atcg::make_scope<Impl>(_ID);
    impl->size     = size;
    impl->capacity = size;
}

VertexBuffer::VertexBuffer(const void* data, size_t size)
{
    glGenBuffers(1, &_ID);
    glBindBuffer(GL_ARRAY_BUFFER, _ID);
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_DYNAMIC_DRAW);

    impl           = atcg::make_scope<Impl>(_ID);
    impl->size     = size;
    impl->capacity = size;
}

VertexBuffer::~VertexBuffer()
{
    unmapPointers();
    impl->deinitResource();
    glDeleteBuffers(1, &_ID);
}

void VertexBuffer::use() const
{
    unmapPointers();
    glBindBuffer(GL_ARRAY_BUFFER, _ID);
}

void VertexBuffer::bindStorage(uint32_t slot) const
{
    unmapPointers();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, slot, _ID);
}

void VertexBuffer::setData(const void* data, size_t size)
{
    unmapPointers();
    glBindBuffer(GL_ARRAY_BUFFER, _ID);
    resize(size);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size, data);
    impl->size = size;
}

void VertexBuffer::resize(std::size_t size)
{
    if(size <= impl->capacity)
    {
        if(impl->resource_ready) { impl->deinitResource(); }
        glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
        impl->capacity = size;
        impl->initResource(_ID);
    }
    impl->size = size;
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

void VertexBuffer::unmapHostPointers() const
{
    impl->unmapResourceHost();
}

void VertexBuffer::unmapDevicePointers() const
{
    impl->unmapResourceDevice();
}

void VertexBuffer::unmapPointers() const
{
    unmapHostPointers();
    unmapDevicePointers();
}

bool VertexBuffer::isDeviceMapped() const
{
    return impl->mapped_device;
}

bool VertexBuffer::isHostMapped() const
{
    return impl->mapped_host;
}

std::size_t VertexBuffer::size() const
{
    return impl->size;
}

std::size_t VertexBuffer::capacity() const
{
    return impl->capacity;
}

IndexBuffer::IndexBuffer() : VertexBuffer() {}

IndexBuffer::IndexBuffer(const uint32_t* indices, size_t count) : VertexBuffer((void*)indices, count * sizeof(uint32_t))
{
}

IndexBuffer::IndexBuffer(size_t count) : VertexBuffer(count * sizeof(uint32_t)) {}

IndexBuffer::~IndexBuffer() {}

void IndexBuffer::use() const
{
    unmapPointers();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ID);
}

void IndexBuffer::setData(const uint32_t* data, size_t count)
{
    unmapPointers();
    glBindBuffer(GL_ARRAY_BUFFER, _ID);
    resize(count * sizeof(uint32_t));
    glBufferSubData(GL_ARRAY_BUFFER, 0, count * sizeof(uint32_t), data);
}

}    // namespace atcg