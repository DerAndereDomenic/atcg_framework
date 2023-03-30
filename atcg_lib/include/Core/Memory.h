#pragma once

#include <Core/CUDA.h>
#include <memory>
#include <cstring>

namespace atcg
{

/**
 * @brief This class handles a buffer.
 * It is device agnostic for cuda code. It should only be used if a
 * buffer should be useable on both backends (CPU/CUDA).
 * If you only require CPU usage, you should use std containers
 */
template<typename T>
class MemoryContainer
{
public:
    /**
     * @brief Create a memory container
     */
    MemoryContainer() = default;

    /**
     * @brief Destructor
     */
    ~MemoryContainer() { destroy(); }

    /**
     * @brief Create a memory container of size n.
     * Only allocates new memory if n < capacity of the container
     *
     * @param n The number of elements
     * @return If the allocation was succesfull
     */
    bool create(std::size_t n)
    {
        if(n <= _capacity)
        {
            _size = n;
            return true;
        }

        destroy();

        if(n <= 0) { return false; }

        std::size_t buffer_size = n * sizeof(T);

#ifdef ATCG_CUDA_BACKEND
        cudaError_t status = cudaMalloc((void**)&_buffer, buffer_size);
        if(status != cudaSuccess) { return false; }
#else
        _buffer = static_cast<T*>(std::malloc(buffer_size));
        if(!_buffer) { return false; }
#endif
        _size     = n;
        _capacity = n;
        return true;
    }

    void destroy()
    {
        if(_buffer)
        {
#ifdef ATCG_CUDA_BACKEND
            cudaFree(_buffer);
#else
            std::free(_buffer);
#endif
        }
        _size     = 0;
        _capacity = 0;
    }

    /**
     * @brief Upload data onto the device.
     * @note For CPU backend, a memory copy is performed
     *
     * @param host_data The data to upload
     * @param n The number of elements
     */
    void upload(const T* host_data, std::size_t n)
    {
        if(!create()) return;
        size_t buffer_size = sizeof(T) * n;
        _size              = n;
#ifdef ATCG_CUDA_BACKEND
        cudaMemcpy((void*)_buffer, (void*)host_data, buffer_size, cudaMemcpyHostToDevice);
#else
        std::memcpy(_buffer, host_data, buffer_size);
#endif
    }

    /**
     * @brief Download data from the device.
     * @note For CPU backed, a memory copy is performed
     *
     * @param host_data Pointer to where the data should be written to
     * @param n The number of elements
     */
    void download(const T* host_data, std::size_t n)
    {
        size_t buffer_size = sizeof(T) * n;
#ifdef ATCG_CUDA_BACKEND
        cudaMemcpy((void*)host_data, (void*)_buffer, buffer_size, cudaMemcpyDeviceToHost);
#else
        std::memcpy(host_data, _buffer, buffer_size);
#endif
    }

    /**
     * @brief The size of the container, i.e., the number of elements
     *
     * @return The size
     */
    std::size_t size() const { return _size; }

    /**
     * @brief The capacity of the container.
     * May be larger than its size
     *
     * @return The capacity
     */
    std::size_t capacity() const { return _capacity; }

    /**
     * @brief Get the raw (device) buffer.
     *
     * @return The device buffer
     */
    T* get() const { return _buffer; }

private:
    std::size_t _capacity = 0;
    std::size_t _size     = 0;
    T* _buffer            = nullptr;
};

/**
 * @brief These memory objects currently only replace std::shared_ptr and std::unique_ptr.
 */

template<typename T>
using scope_ptr = std::unique_ptr<T>;
template<typename T>
using ref_ptr = std::shared_ptr<T>;

template<typename T, typename... Args>
constexpr scope_ptr<T> make_scope(Args&&... args)
{
    return std::make_unique<T>(std::forward<Args>(args)...);
}


template<typename T, typename... Args>
constexpr ref_ptr<T> make_ref(Args&&... args)
{
    return std::make_shared<T>(std::forward<Args>(args)...);
}

/**
 * @brief A device buffer with managed memory
 */
template<typename T>
class DeviceBuffer
{
public:
    /**
     * @brief Constructor
     *
     */
    DeviceBuffer() { _container = make_ref<MemoryContainer<T>>(); }

    /**
     * @brief Constructor
     *
     * @param n Size of the buffer
     *
     */
    DeviceBuffer(std::size_t n)
    {
        _container = make_ref<MemoryContainer<T>>();
        create(n);
    }

    /**
     * @brief Destructor
     *
     */
    ~DeviceBuffer()
    {
        if(_container) _container.reset();
    }

    /**
     * @brief Destroy the underlying buffer, i.e., free device memory
     *
     */
    void destroy()
    {
        if(_container) _container->destroy();
    }

    /**
     * @brief Get the data
     *
     * @return Data pointer on the device
     *
     */
    T* get() const { return _container->get(); }

    /**
     * @brief The size of the container, i.e., the number of elements
     *
     * @return The size
     */
    std::size_t size() const { return _container->size(); }

    /**
     * @brief The capacity of the container.
     * May be larger than its size
     *
     * @return The capacity
     */
    std::size_t capacity() const { return _container->capacity(); }

    /**
     * @brief Create a buffer of size n.
     *
     * @param n The number of elements
     * @return If the allocation was successful
     */
    bool create(std::size_t n) { return _container->create(n); }

    /**
     * @brief Upload the data.
     * Copies buffer->size() many elements.
     *
     * @param host_data The source host buffer
     */
    void upload(const T* host_data) { _container->upload(host_data, _container->size()); }

    /**
     * @brief Upload the data.
     *
     * @param host_data The source host buffer
     * @param n The number of elements
     */
    void upload(const T* host_data, size_t n) { _container->upload(host_data, n); }

    /**
     * @brief Download the data.
     * Copies buffer->size() many elements.
     *
     * @param host_data The target host buffer
     */
    void download(const T* host_data) { _container->download(host_data, _container->size()); }

    /**
     * @brief Download the data.
     *
     * @param host_data The target host buffer
     * @param n The number of elements
     */
    void download(const T* host_data, size_t n) { _container->download(host_data, n); }

private:
    ref_ptr<MemoryContainer<T>> _container;
};

}    // namespace atcg