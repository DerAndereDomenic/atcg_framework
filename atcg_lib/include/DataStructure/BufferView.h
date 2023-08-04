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
     * @param stride The strde between consecutive elements in bytes
     */
    ATCG_HOST_DEVICE
    BufferView(uint8_t* ptr, std::size_t stride = sizeof(T)) : _ptr(ptr), _stride(stride) {}

    /**
     * @brief Get the element at a given index
     * @param index The index
     *
     * @return The element at that index
     */
    ATCG_HOST_DEVICE
    T& operator[](uint32_t index) { return *(T*)(_ptr + index * _stride); }

    /**
     * @brief Pre-Increment the pointer
     */
    ATCG_HOST_DEVICE
    BufferView<T>& operator++()
    {
        ++_idx;
        return *this;
    }

    /**
     * @brief Post-Increment the pointer
     */
    ATCG_HOST_DEVICE
    BufferView<T> operator++(int)
    {
        BufferView<T> result(*this);
        ++(*this);
        return result;
    }

    /**
     * @brief Get the current element of the pointer
     */
    ATCG_HOST_DEVICE
    T operator*() { return *(T*)(_ptr + _idx * _stride); }

private:
    uint8_t* _ptr       = nullptr;
    std::size_t _idx    = 0;
    std::size_t _stride = sizeof(T);
};
}    // namespace atcg