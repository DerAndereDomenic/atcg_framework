#pragma once

#include <Core/Memory.h>

#include <cstdint>
#include <string>
#include <vector>

namespace atcg
{
/**
 * @brief This enum holds the possible vertex attributes that can be passed to a shader via a vertex buffer
 *
 */
enum class ShaderDataType
{
    None = 0,
    Float,
    Float2,
    Float3,
    Float4,
    Mat3,
    Mat4,
    Int,
    Int2,
    Int3,
    Int4,
    Bool
};

/**
 * @brief Get the size of an element in byte
 *
 * @param type The Data type
 * @return uint32_t The size in bytes
 */
static uint32_t ShaderDataTypeSize(ShaderDataType type)
{
    switch(type)
    {
        case ShaderDataType::Float:
            return 4;
        case ShaderDataType::Float2:
            return 4 * 2;
        case ShaderDataType::Float3:
            return 4 * 3;
        case ShaderDataType::Float4:
            return 4 * 4;
        case ShaderDataType::Mat3:
            return 4 * 3 * 3;
        case ShaderDataType::Mat4:
            return 4 * 4 * 4;
        case ShaderDataType::Int:
            return 4;
        case ShaderDataType::Int2:
            return 4 * 2;
        case ShaderDataType::Int3:
            return 4 * 3;
        case ShaderDataType::Int4:
            return 4 * 4;
        case ShaderDataType::Bool:
            return 1;
        default:
            return 0;
    }

    return 0;
};

/**
 * @brief A class to model an element inside the buffer
 *
 */
struct BufferElement
{
    std::string name;
    ShaderDataType type;
    uint32_t size;
    size_t offset;
    bool normalized;

    BufferElement() = default;

    BufferElement(ShaderDataType type, const std::string& name, bool normalized = false)
        : name(name),
          type(type),
          size(ShaderDataTypeSize(type)),
          offset(0),
          normalized(normalized)
    {
    }

    uint32_t getComponentCount() const
    {
        switch(type)
        {
            case ShaderDataType::Float:
                return 1;
            case ShaderDataType::Float2:
                return 2;
            case ShaderDataType::Float3:
                return 3;
            case ShaderDataType::Float4:
                return 4;
            case ShaderDataType::Mat3:
                return 3;    // 3* float3
            case ShaderDataType::Mat4:
                return 4;    // 4* float4
            case ShaderDataType::Int:
                return 1;
            case ShaderDataType::Int2:
                return 2;
            case ShaderDataType::Int3:
                return 3;
            case ShaderDataType::Int4:
                return 4;
            case ShaderDataType::Bool:
                return 1;
            default:
                return 0;
        }

        return 0;
    }
};

/**
 * @brief The layout of the buffer.
 * Basically a collection of BufferElements
 */
class BufferLayout
{
public:
    BufferLayout() {}

    BufferLayout(std::initializer_list<BufferElement> elements) : _elements(elements) { calculateOffsetAndStride(); }

    uint32_t getStride() const { return _stride; }

    const std::vector<BufferElement>& getElements() const { return _elements; }

    std::vector<BufferElement>::iterator begin() { return _elements.begin(); }
    std::vector<BufferElement>::iterator end() { return _elements.end(); }
    std::vector<BufferElement>::const_iterator begin() const { return _elements.begin(); }
    std::vector<BufferElement>::const_iterator end() const { return _elements.end(); }

private:
    void calculateOffsetAndStride()
    {
        size_t offset = 0;
        _stride       = 0;
        for(auto& element: _elements)
        {
            element.offset = offset;
            offset += element.size;
            _stride += element.size;
        }
    }

private:
    std::vector<BufferElement> _elements;
    uint32_t _stride = 0;
};

/**
 * @brief A class to model a vertex buffer
 */
class VertexBuffer
{
public:
    /**
     * @brief Construct a new Vertex Buffer object
     */
    VertexBuffer();

    /**
     * @brief Construct a new empty Vertex Buffer object
     *
     * @param size The size of bytes allocated on the GPU
     */
    VertexBuffer(size_t size);

    /**
     * @brief Construct a new Vertex Buffer object
     *
     * @param data The data to fill the buffer with
     * @param size The size in bytes
     */
    VertexBuffer(const void* data, size_t size);

    /**
     * @brief Destroy the Vertex Buffer object
     */
    ~VertexBuffer();

    /**
     * @brief Use this vbo
     *
     * @note Invalidades the device pointer obtained by "getDevicePointer"
     */
    void use() const;

    /**
     * @brief Bind the buffer as Shader Storage Buffer
     *
     * @note Invalidades the device pointer obtained by "getDevicePointer"
     *
     * @param slot The slot to bind to
     */
    void bindStorage(uint32_t slot = 0) const;

    /**
     * @brief Set the Data of the buffer. The data gets copied if size < capacity(). Otherwise a buffer with capacity()
     * = size will be allocated.
     *
     * @note Invalidades the device pointer obtained by "getDevicePointer"
     *
     * @param data The data
     * @param size The size in bytes
     */
    void setData(const void* data, size_t size);

    /**
     * @brief Resize the buffer.
     * If size <= capacity(), there will be no new allocation but excessive memory is not freed.
     *
     * @param size The new size in bytes
     */
    void resize(std::size_t size);

    /**
     * @brief Get the underlying data as a device pointer.
     * This only returns a valid device pointer if the CUDA backend is enabled. Otherwise this will return the buffer
     * mapped to host space.
     *
     * @note This function should be called every frame and the pointer should not be cached by the application. OpenGL
     * is allowed to move buffers in memory. Therefore, the pointer might no longer be valid. The underlying resource
     * gets mapped and unmapped automatically. Every call to "use", "bindStorage" or "setData" invalidates the pointer.
     * If the buffer does not get explicitly binded again (because a Vertex array for example only points to this
     * buffer), the client has to manually unmap the pointers using "unmapPointers()" before any further rendering calls
     * can be done.
     *
     * @return The pointer
     */
    void* getDevicePointer() const;

    /**
     * @brief Get the underlying data as a device pointer.
     * This only returns a valid device pointer if the CUDA backend is enabled. Otherwise this will return the buffer
     * mapped to host space.
     *
     * @note This function should be called every frame and the pointer should not be cached by the application. OpenGL
     * is allowed to move buffers in memory. Therefore, the pointer might no longer be valid. The underlying resource
     * gets mapped and unmapped automatically. Every call to "use", "bindStorage" or "setData" invalidates the pointer.
     * If the buffer does not get explicitly binded again (because a Vertex array for example only points to this
     * buffer), the client has to manually unmap the pointers using "unmapPointers()" before any further rendering calls
     * can be done..
     *
     * \tparam T The type on how to interpret the data
     *
     * @return The pointer
     */
    template<typename T>
    inline T* getDevicePointer() const
    {
        return reinterpret_cast<T*>(getDevicePointer());
    }

    /**
     * @brief Get the underlying data as a host pointer.
     *
     * @note This function should be called every frame and the pointer should not be cached by the application. OpenGL
     * is allowed to move buffers in memory. Therefore, the pointer might no longer be valid. The underlying resource
     * gets mapped and unmapped automatically. Every call to "use", "bindStorage" or "setData" invalidates the pointer.
     * If the buffer does not get explicitly binded again (because a Vertex array for example only points to this
     * buffer), the client has to manually unmap the pointers using "unmapPointers()" before any further rendering calls
     * can be done.
     *
     * @return The pointer
     */
    void* getHostPointer() const;

    /**
     * @brief Get the underlying data as a host pointer.
     *
     * @note This function should be called every frame and the pointer should not be cached by the application. OpenGL
     * is allowed to move buffers in memory. Therefore, the pointer might no longer be valid. The underlying resource
     * gets mapped and unmapped automatically. Every call to "use", "bindStorage" or "setData" invalidates the pointer.
     * If the buffer does not get explicitly binded again (because a Vertex array for example only points to this
     * buffer), the client has to manually unmap the pointers using "unmapPointers()" before any further rendering calls
     * can be done.
     *
     * \tparam T The type on how to interpret the data
     *
     * @return The pointer
     */
    template<typename T>
    inline T* getHostPointer() const
    {
        return reinterpret_cast<T*>(getHostPointer());
    }

    /**
     * @brief Unmaps and invalidates all host pointers used by the application
     */
    void unmapHostPointers() const;

    /**
     * @brief Unmaps and invalidates all device pointers used by the application
     */
    void unmapDevicePointers() const;

    /**
     * @brief Unmaps and invalidates all mapped pointers used by the application.
     */
    void unmapPointers() const;

    /**
     * @brief Check if the device pointer is mapped (valid).
     * @return True if the pointer is valid
     */
    bool isDeviceMapped() const;

    /**
     * @brief Check if the host pointer is mapped (valid).
     * @return True if the host is valid
     */
    bool isHostMapped() const;

    /**
     * @brief Get the Layout
     *
     * @return const BufferLayout& The layout of the buffer
     */
    inline const BufferLayout& getLayout() const { return _layout; }

    /**
     * @brief Set the Layout
     *
     * @param layout The layout
     */
    inline void setLayout(const BufferLayout& layout) { _layout = layout; }

    /**
     * @brief Get the size in use in bytes
     *
     * @return The size
     */
    std::size_t size() const;

    /**
     * @brief Get the capacity of the buffer in bytes
     *
     * @return The capacity
     */
    std::size_t capacity() const;

    /**
     * @brief Get the internal rendering ID of this buffer
     *
     * @return The renderer ID
     */
    inline uint32_t ID() const { return _ID; }

protected:
    class Impl;
    atcg::scope_ptr<Impl> impl;
    uint32_t _ID;
    BufferLayout _layout;
};

/**
 * @brief A class to model an IndexBuffer.
 * An index buffer is just a vertex buffer with uint32_t and a different binding
 */
class IndexBuffer : public VertexBuffer
{
public:
    /**
     * @brief Construct a new Index Buffer object
     */
    IndexBuffer();

    /**
     * @brief Construct a new Index Buffer object
     *
     * @param data The data
     * @param count The size
     */
    IndexBuffer(const uint32_t* indices, size_t count);

    /**
     * @brief Construct a new Index Buffer object
     *
     * @param count The size
     */
    IndexBuffer(size_t count);

    /**
     * @brief Destroy the Index Buffer object
     */
    ~IndexBuffer();

    /**
     * @brief Use this ibo
     */
    void use() const;

    /**
     * @brief Set the Data of the buffer. The data gets copied if size < capacity(). Otherwise a buffer with capacity()
     * = size will be allocated.
     *
     * @param data The data
     * @param size The count
     */
    void setData(const uint32_t* data, size_t count);

    /**
     * @brief Get the Count of objects
     *
     * @return uint32_t The count
     */
    inline size_t getCount() const { return size() / sizeof(uint32_t); }

private:
};
}    // namespace atcg