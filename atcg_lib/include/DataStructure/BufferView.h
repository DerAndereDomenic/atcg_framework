#pragma once

namespace atcg
{

/**
 * @brief A class to model a buffer view
 *
 * @tparam T The type
 */
template<typename T>
class BufferView
{
public:
    /**
     * @brief Default constructor
     */
    BufferView() = default;

    /**
     * @brief Construct a new BufferView.
     * This is a non-owning reference, i.e., this buffer is valid as long as the given pointer is valid.
     *
     * @param ptr The pointer
     * @param size The size in bytes
     * @param stride The strde between consecutive elements in bytes
     */
    template<typename Q>
    ATCG_HOST_DEVICE BufferView(Q* ptr, uint32_t size, uint32_t stride = sizeof(T))
        : _ptr((uint8_t*)ptr),
          _size(size),
          _stride(stride)
    {
    }

    /**
     * @brief Get the element at a given index
     * @param index The index
     *
     * @return The element at that index
     */
    ATCG_HOST_DEVICE ATCG_INLINE T& operator[](uint32_t index) { return *(T*)(_ptr + index * _stride); }

    /**
     * @brief Pre-Increment the pointer
     */
    ATCG_HOST_DEVICE ATCG_INLINE BufferView<T>& operator++()
    {
        ++_idx;
        return *this;
    }

    /**
     * @brief Post-Increment the pointer
     */
    ATCG_HOST_DEVICE ATCG_INLINE BufferView<T> operator++(int)
    {
        BufferView<T> result(*this);
        ++(*this);
        return result;
    }

    /**
     * @brief Get the current element of the pointer
     */
    ATCG_HOST_DEVICE ATCG_INLINE T operator*() { return *(T*)(_ptr + _idx * _stride); }

    /**
     * @brief Get the number of elements
     *
     * @return The number of elements
     */
    ATCG_HOST_DEVICE ATCG_INLINE uint32_t numel() const { return _size / _stride; }

    /**
     * @brief Get the size in bytes
     *
     * @return The size in bytes
     */
    ATCG_HOST_DEVICE ATCG_INLINE uint32_t size() const { return _size; }

    /**
     * @brief Get the stride in bytes
     *
     * @return The stride in bytes
     */
    ATCG_HOST_DEVICE ATCG_INLINE uint32_t stride() const { return _stride; }

    /**
     * @brief Reset to the base pointer
     */
    ATCG_HOST_DEVICE ATCG_INLINE void reset() { _idx = 0; }

private:
    uint8_t* _ptr    = nullptr;
    uint32_t _idx    = 0;
    uint32_t _size   = 0;
    uint32_t _stride = sizeof(T);
};
}    // namespace atcg