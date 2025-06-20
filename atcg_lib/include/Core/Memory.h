#pragma once

#include <Core/CUDA.h>
#include <Core/Log.h>

#include <memory>
#include <cstring>
#include <cstdlib>

namespace atcg
{

/**
 * @brief An allocator that handles memory allocated on the host
 *
 */
struct host_allocator
{
    /**
     * @brief Allocate n bytes on the host
     * @param n The number to allocate in bytes
     *
     * @return A pointer to the allocated memory
     */
    void* allocate(std::size_t n)
    {
        bytes_allocated += n;
        return std::malloc(n);
    }

    /**
     * @brief Deletes the memory associated with the pointer
     * @param ptr The memory that should be deleted
     * @param n The number of bytes that are deleted
     */
    void deallocate(void* ptr, std::size_t n)
    {
        bytes_deallocated += n;
        std::free(ptr);
    }

    /**
     * @brief Performs a memcopy from host to host
     * @param dst The host destination pointer
     * @param src The host source pointer
     * @param n The number of bytes to copy
     */
    void memcpy_host2host(void* dst, const void* src, std::size_t n) { std::memcpy(dst, src, n); }

    /**
     * @brief Performs a memcopy from device to host
     * @param dst The device destination pointer
     * @param src The host source pointer
     * @param n The number of bytes to copy
     */
    void memcpy_dev2host(void* dst, const void* src, std::size_t n) { memcpy_host2host(dst, src, n); }

    /**
     * @brief Performs a memcopy from host to device
     * @param dst The device destination pointer
     * @param src The host source pointer
     * @param n The number of bytes to copy
     */
    void memcpy_host2dev(void* dst, const void* src, std::size_t n) { memcpy_host2host(dst, src, n); }

    /**
     * @brief Performs a memcopy from device to device
     * @param dst The device destination pointer
     * @param src The device source pointer
     * @param n The number of bytes to copy
     */
    void memcpy_dev2dev(void* dst, const void* src, std::size_t n) { memcpy_host2host(dst, src, n); }

    static std::size_t bytes_allocated;
    static std::size_t bytes_deallocated;
};
#ifdef ATCG_CUDA_BACKEND
/**
 * @brief An allocator that handles memory allocated on the device
 *
 */
struct device_allocator
{
    /**
     * @brief Allocate n bytes on the device
     * @param n The number to allocate in bytes
     *
     * @return A pointer to the allocated memory
     */
    void* allocate(std::size_t n)
    {
        bytes_allocated += n;
        void* ptr;
        CUDA_SAFE_CALL(cudaMalloc((void**)&ptr, n));
        return ptr;
    }

    /**
     * @brief Deletes the memory associated with the pointer
     * @param ptr The memory that should be deleted
     * @param n The number of bytes that are deleted
     */
    void deallocate(void* ptr, std::size_t n)
    {
        bytes_deallocated += n;
        cudaFree(ptr);
    }

    /**
     * @brief Performs a memcopy from host to host
     * @param dst The host destination pointer
     * @param src The host source pointer
     * @param n The number of bytes to copy
     */
    void memcpy_host2host(void* dst, const void* src, std::size_t n)
    {
        CUDA_SAFE_CALL(cudaMemcpy(dst, src, n, cudaMemcpyHostToHost));
    }

    /**
     * @brief Performs a memcopy from device to host
     * @param dst The device destination pointer
     * @param src The host source pointer
     * @param n The number of bytes to copy
     */
    void memcpy_dev2host(void* dst, const void* src, std::size_t n)
    {
        CUDA_SAFE_CALL(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost));
    }

    /**
     * @brief Performs a memcopy from host to device
     * @param dst The device destination pointer
     * @param src The host source pointer
     * @param n The number of bytes to copy
     */
    void memcpy_host2dev(void* dst, const void* src, std::size_t n)
    {
        CUDA_SAFE_CALL(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice));
    }

    /**
     * @brief Performs a memcopy from device to device
     * @param dst The device destination pointer
     * @param src The device source pointer
     * @param n The number of bytes to copy
     */
    void memcpy_dev2dev(void* dst, const void* src, std::size_t n)
    {
        CUDA_SAFE_CALL(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToDevice));
    }

    static std::size_t bytes_allocated;
    static std::size_t bytes_deallocated;
};
#else
using device_allocator = host_allocator;
#endif

/**
 * @brief This class handles a buffer.
 * It is device agnostic for cuda code.
 */
template<class allocator>
class MemoryContainer
{
public:
    /**
     * @brief Create a memory container
     */
    MemoryContainer() = default;

    /**
     * @brief Create a memory container that gets unique ownership of the supplied pointer.
     * The memory will be freed if this object is destroyed.
     * @param ptr The pointer to take ownership from
     * @param size The size of the underlying memory in bytes
     */
    MemoryContainer(void* ptr, std::size_t size) : _buffer(ptr), _size(size), _capacity(size) {}

    MemoryContainer(MemoryContainer&) = delete;

    MemoryContainer operator=(MemoryContainer&) = delete;

    /**
     * @brief Destructor, frees the memory owned by this buffer
     */
    ~MemoryContainer() { destroy(); }

    /**
     * @brief Create a memory container of size n (bytes).
     * Only allocates new memory if n < capacity of the container
     *
     * @param n The memory to allocate in bytes
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

        if(n <= 0)
        {
            return false;
        }

        allocator alloc;
        _buffer = alloc.allocate(n);
        if(!_buffer)
        {
            return false;
        }
        _size     = n;
        _capacity = n;
        return true;
    }

    /**
     * @brief Upload data onto the device.
     * Will allocate n bytes if the buffer is too small.
     * @note For CPU backend, a memory copy is performed
     *
     * @param host_data The data to upload
     * @param n The size to be uploaded in bytes
     */
    void upload(const void* host_data, std::size_t n)
    {
        if(!create(n)) return;
        _size = n;
        allocator alloc;
        alloc.memcpy_host2dev(_buffer, host_data, n);
    }

    /**
     * @brief Download data from the device.
     * @note For CPU backed, a memory copy is performed
     *
     * @param host_data Pointer to where the data should be written to
     * @param n The size to be copied in bytes
     */
    void download(void* host_data, std::size_t n)
    {
        allocator alloc;
        alloc.memcpy_dev2host(host_data, _buffer, n);
    }

    /**
     * @brief Copy data on the device
     *
     * @param data The data to copy from
     * @param n The size to be copied in bytes
     *
     */
    void copy(const void* src_data, std::size_t n)
    {
        allocator alloc;
        alloc.memcpy_dev2dev(_buffer, src_data, n);
    }

    /**
     * @brief The size of the container, i.e., bytes of the buffer
     *
     * @return The size in bytes
     */
    std::size_t size() const { return _size; }

    /**
     * @brief The capacity of the container in bytes.
     * May be larger than its size
     *
     * @return The capacity in bytes
     */
    std::size_t capacity() const { return _capacity; }

    /**
     * @brief Get the raw (device) buffer.
     *
     * @return The device buffer
     */
    void* get() const { return _buffer; }

private:
    /**
     * @brief Destroys the buffer and the associated resources
     */
    void destroy()
    {
        if(_buffer)
        {
            allocator alloc;
            alloc.deallocate(_buffer, _capacity);
            _size     = 0;
            _capacity = 0;
            _buffer   = nullptr;
        }
    }

    void* _buffer         = nullptr;
    std::size_t _size     = 0;
    std::size_t _capacity = 0;
};

template<typename T, typename allocator>
struct ContainerDeleter
{
    void operator()(MemoryContainer<allocator>* container)
    {
        // We are the last object that owns the memory -> call destructor first
        if(std::is_same<allocator, host_allocator>::value && container->get() != nullptr)
        {
            T* obj = static_cast<T*>(container->get());
            obj->~T();
        }
        delete container;
    }
};

/**
 * @brief A device buffer with managed memory
 */
template<typename T, typename allocator = host_allocator>
class DevicePointer
{
public:
    /**
     * @brief Constructor
     *
     */
    DevicePointer()
    {
        _container = std::shared_ptr<MemoryContainer<allocator>>(new MemoryContainer<allocator>(),
                                                                 ContainerDeleter<T, allocator>());
    }

    /**
     * @brief Assign nullptr
     */
    constexpr DevicePointer(std::nullptr_t) noexcept
    {
        _container = std::shared_ptr<MemoryContainer<allocator>>(new MemoryContainer<allocator>(),
                                                                 ContainerDeleter<T, allocator>());
    }

    /**
     * @brief Construct a device buffer from a pointer.
     * The pointer already has to be on the right device.
     * The memory will be assumed to be exactly the size of on object (sizeof(T))
     *
     * @param ptr The pointer
     */
    DevicePointer(T* ptr)
    {
        MemoryContainer<allocator>* obj = new MemoryContainer<allocator>(ptr, sizeof(T));

        _container = std::shared_ptr<MemoryContainer<allocator>>(obj, ContainerDeleter<T, allocator>());
    }

    /**
     * @brief Destructor
     */
    ~DevicePointer() {}

    /**
     * @brief Get the data
     *
     * @return Data pointer on the device
     *
     */
    T* get() const { return reinterpret_cast<T*>(_container->get()); }

    /**
     * @brief Get the data
     *
     * @return Data pointer on the device
     *
     */
    T* operator*() const { return reinterpret_cast<T*>(_container->get()); }

    /**
     * @brief Get the data
     *
     * @return Data pointer on the device
     *
     */
    T* operator->() const { return reinterpret_cast<T*>(_container->get()); }

    /**
     * @brief operator bool
     * @return True if this buffer manages any memory
     */
    explicit operator bool() const { return _container && _container->get() != nullptr; }

    /**
     * @brief Convert between different types that are holded by this buffer.
     * After this operation, the use count of the underlying shared_ptr will be 2.
     * Both, the original and the newly constructed object will handle the same memory
     * just interpreted differently. This means that MemoryBuffer<float> a = MemoryBuffer<int>(5)
     * will not cast 5 to 5.0f but will just interpet the bits of 5 as float.
     * @tparam The type of the other object
     */
    template<typename U>
    operator DevicePointer<U, allocator>() const
    {
        DevicePointer<U> ptr;
        ptr._container = _container;
        return ptr;
    }

    /**
     * @brief Upload the data.
     * Copies buffer->size() many elements.
     *
     * @param host_data The source host buffer
     */
    void upload(T* host_data)
    {
        _container->upload(reinterpret_cast<void*>(host_data), std::max(sizeof(T), _container->size()));
    }

    /**
     * @brief Download the data.
     * Copies buffer->size() many elements.
     *
     * @param host_data The target host buffer
     */
    void download(T* host_data)
    {
        _container->download(reinterpret_cast<void*>(host_data), std::max(sizeof(T), _container->size()));
    }

    /**
     * @brief Copy to the buffer
     * Copies buffer->size() many elements.
     *
     * @param src_data The source buffer
     */
    void copy(T* src_data)
    {
        _container->copy(reinterpret_cast<void*>(src_data), std::max(sizeof(T), _container->size()));
    }

    /**
     * @brief Copy to the buffer.
     * This allocates src.size() many objects. *this has to be the correct size beforehand
     *
     * @param src The source pointer
     */
    void copy(const DevicePointer<T, allocator>& src) { _container->copy(src.get(), src.size() * sizeof(T)); }

    /**
     * @brief Reset this shared_ptr, i.e. decrement the ref counter of the underlying shared_ptr
     *
     */
    void reset() noexcept { _container.reset(); }

    /**
     * @brief Reset this buffer and make it manage different memory.
     * @tparam U The type of the new data
     * @param ptr Pointer to the new object. Size is assumed to be sizeof(U)
     */
    template<typename U>
    void reset(U* ptr)
    {
        MemoryContainer<allocator>* obj = new MemoryContainer<allocator>(ptr, sizeof(U));

        _container = std::shared_ptr<MemoryContainer<allocator>>(obj, ContainerDeleter<T, allocator>());
    }

    /**
     * @brief Get the number of pointers that point to the memory owned by this object
     *
     * @return The use count
     */
    int64_t use_count() const { return _container.use_count(); }

private:
    template<typename U, typename _allocator>
    friend class DevicePointer;

    std::shared_ptr<MemoryContainer<allocator>> _container;
};

/**
 * @brief A device buffer with managed memory
 */
template<typename T, typename allocator = host_allocator>
class MemoryBuffer
{
public:
    /**
     * @brief Constructor
     *
     */
    MemoryBuffer()
    {
        _container = std::shared_ptr<MemoryContainer<allocator>>(new MemoryContainer<allocator>(),
                                                                 ContainerDeleter<T, allocator>());
    }

    /**
     * @brief Constructor
     *
     * @param n Size of the buffer (number of elements)
     *
     */
    MemoryBuffer(std::size_t n)
    {
        _container = std::shared_ptr<MemoryContainer<allocator>>(new MemoryContainer<allocator>(),
                                                                 ContainerDeleter<T, allocator>());
        create(n);
    }

    /**
     * @brief Assign nullptr
     */
    constexpr MemoryBuffer(std::nullptr_t) noexcept
    {
        _container = std::shared_ptr<MemoryContainer<allocator>>(new MemoryContainer<allocator>(),
                                                                 ContainerDeleter<T, allocator>());
    }

    /**
     * @brief Construct a device buffer from a pointer.
     * The pointer already has to be on the right device.
     * The memory will be assumed to be exactly the size of on object (no arrays)
     *
     * @param ptr The pointer
     */
    MemoryBuffer(T* ptr)
    {
        MemoryContainer<allocator>* obj = new MemoryContainer<allocator>(ptr, sizeof(T));

        _container = std::shared_ptr<MemoryContainer<allocator>>(obj, ContainerDeleter<T, allocator>());
    }

    /**
     * @brief Construct a device buffer from a pointer.
     * The pointer already has to be on the right device.
     *
     * @param ptr The pointer
     * @param size The number of elements
     */
    MemoryBuffer(T* ptr, std::size_t size)
    {
        MemoryContainer<allocator>* obj = new MemoryContainer<allocator>(ptr, size * sizeof(T));

        _container = std::shared_ptr<MemoryContainer<allocator>>(obj, ContainerDeleter<T, allocator>());
    }

    /**
     * @brief Destructor
     */
    ~MemoryBuffer() {}

    /**
     * @brief Get the data
     *
     * @return Data pointer on the device
     *
     */
    T* get() const { return reinterpret_cast<T*>(_container->get()); }

    /**
     * @brief Get the data
     *
     * @return Data pointer on the device
     *
     */
    T* operator*() const { return reinterpret_cast<T*>(_container->get()); }

    /**
     * @brief Get the data
     *
     * @return Data pointer on the device
     *
     */
    T* operator->() const { return reinterpret_cast<T*>(_container->get()); }

    /**
     * @brief operator bool
     * @return True if this buffer manages any memory
     */
    explicit operator bool() const { return _container && _container->get() != nullptr; }

    /**
     * @brief Convert between different types that are holded by this buffer.
     * After this operation, the use count of the underlying shared_ptr will be 2.
     * Both, the original and the newly constructed object will handle the same memory
     * just interpreted differently. This means that MemoryBuffer<float> a = MemoryBuffer<int>(5)
     * will not cast 5 to 5.0f but will just interpet the bits of 5 as float.
     * @tparam The type of the other object
     */
    template<typename U>
    operator MemoryBuffer<U, allocator>() const
    {
        MemoryBuffer<U> ptr;
        ptr._container = _container;
        return ptr;
    }

    /**
     * @brief The size of the container, i.e., the number of elements
     *
     * @return The size
     */
    std::size_t size() const { return _container->size() / sizeof(T); }

    /**
     * @brief The capacity of the container.
     * May be larger than its size
     *
     * @return The capacity
     */
    std::size_t capacity() const { return _container->capacity() / sizeof(T); }

    /**
     * @brief Create a buffer of size n.
     *
     * @param n The number of elements
     * @return If the allocation was successful
     */
    bool create(std::size_t n) { return _container->create(sizeof(T) * n); }

    /**
     * @brief Upload the data.
     * Copies buffer->size() many elements.
     *
     * @param host_data The source host buffer
     */
    void upload(T* host_data) { _container->upload(reinterpret_cast<void*>(host_data), _container->size()); }

    /**
     * @brief Upload the data.
     *
     * @param host_data The source host buffer
     * @param n The number of elements
     */
    void upload(T* host_data, size_t n) { _container->upload(reinterpret_cast<void*>(host_data), sizeof(T) * n); }

    /**
     * @brief Download the data.
     * Copies buffer->size() many elements.
     *
     * @param host_data The target host buffer
     */
    void download(T* host_data) { _container->download(reinterpret_cast<void*>(host_data), _container->size()); }

    /**
     * @brief Download the data.
     *
     * @param host_data The target host buffer
     * @param n The number of elements
     */
    void download(T* host_data, size_t n) { _container->download(reinterpret_cast<void*>(host_data), sizeof(T) * n); }

    /**
     * @brief Copy to the buffer
     * Copies buffer->size() many elements.
     *
     * @param src_data The source buffer
     */
    void copy(T* src_data) { _container->copy(reinterpret_cast<void*>(src_data), _container->size()); }

    /**
     * @brief Copy to the buffer
     *
     * @param src_data The source buffer
     * @param n The number of elements
     */
    void copy(T* src_data, std::size_t n) { _container->copy(reinterpret_cast<void*>(src_data), sizeof(T) * n); }

    /**
     * @brief Copy to the buffer.
     * This allocates src.size() many objects. *this has to be the correct size beforehand
     *
     * @param src The source pointer
     */
    void copy(const MemoryBuffer<T, allocator>& src) { _container->copy(src.get(), src.size() * sizeof(T)); }

    /**
     * @brief Reset this shared_ptr, i.e. decrement the ref counter of the underlying shared_ptr
     *
     */
    void reset() noexcept { _container.reset(); }

    /**
     * @brief Reset this buffer and make it manage different memory.
     * @tparam U The type of the new data
     * @param ptr Pointer to the new object. Size is assumed to be sizeof(U)
     */
    template<typename U>
    void reset(U* ptr)
    {
        MemoryContainer<allocator>* obj = new MemoryContainer<allocator>(ptr, sizeof(U));

        _container = std::shared_ptr<MemoryContainer<allocator>>(obj, ContainerDeleter<T, allocator>());
    }

    /**
     * @brief Reset this buffer and make it manage different memory.
     * @tparam U The type of the new data
     * @param ptr Pointer to the new object.
     * @param size The size
     */
    template<typename U>
    void reset(U* ptr, std::size_t size)
    {
        MemoryContainer<allocator>* obj = new MemoryContainer<allocator>(ptr, size);

        _container = std::shared_ptr<MemoryContainer<allocator>>(obj, ContainerDeleter<T, allocator>());
    }

    /**
     * @brief Get the number of pointers that point to the memory owned by this object
     *
     * @return The use count
     */
    int64_t use_count() const { return _container.use_count(); }

private:
    template<typename U, typename _allocator>
    friend class MemoryBuffer;

    std::shared_ptr<MemoryContainer<allocator>> _container;
};

template<typename T, typename allocator>
bool operator==(const DevicePointer<T, allocator>& p1, const DevicePointer<T, allocator>& p2) noexcept
{
    return p1.get() == p2.get();
}

template<typename T, typename allocator>
bool operator!=(const DevicePointer<T, allocator>& p1, const DevicePointer<T, allocator>& p2) noexcept
{
    return p1.get() != p2.get();
}

template<typename T, typename allocator>
bool operator==(const MemoryBuffer<T, allocator>& p1, const MemoryBuffer<T, allocator>& p2) noexcept
{
    return p1.get() == p2.get();
}

template<typename T, typename allocator>
bool operator!=(const MemoryBuffer<T, allocator>& p1, const MemoryBuffer<T, allocator>& p2) noexcept
{
    return p1.get() != p2.get();
}


template<typename T>
using scope_ptr = std::unique_ptr<T>;
template<typename T>
using ref_ptr = std::shared_ptr<T>;
template<typename T>
using dref_ptr = DevicePointer<T, device_allocator>;
template<typename T>
using HostBuffer = MemoryBuffer<T, host_allocator>;
template<typename T>
using DeviceBuffer = MemoryBuffer<T, device_allocator>;

template<typename T, typename... Args>
constexpr scope_ptr<T> make_scope(Args&&... args)
{
    return std::make_unique<T>(std::forward<Args>(args)...);
}

/**
 * @brief Construct an object that is managed by the ref_ptr.
 * This only constructs host objects. For device objects, use the explicit constructor
 * by allocating a buffer of a given size or to pass ownership of a already allocated buffer.
 * @tparam T The type that should be managed
 * @tparam Args Constructor arguments of the object
 * @param args Constructor arguments of the object
 *
 * @return The buffer
 */
template<typename T, typename... Args>
constexpr ref_ptr<T> make_ref(Args&&... args)
{
    // host_allocator alloc;
    // T* buffer = static_cast<T*>(alloc.allocate(sizeof(T)));

    // T* obj = new(buffer) T(std::forward<Args>(args)...);

    // ref_ptr<T> ptr(buffer);
    // return ptr;
    return std::make_shared<T>(std::forward<Args>(args)...);
}

}    // namespace atcg