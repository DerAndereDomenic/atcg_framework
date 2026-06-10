#pragma once

#include <Core/Platform.h>
#include <Core/CUDA.h>

namespace atcg
{
/**
 * @brief Select one of the options
 * @tparam T The type of the options
 * @param condition The condition to select the option
 * @param option_true The option to select if the condition is true
 * @param option_false The option to select if the condition is false
 *
 * @return The selected option
 */
template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T select(bool condition, const T& option_true, const T& option_false)
{
    return condition ? option_true : option_false;
}

namespace Math
{
/**
 * @brief Map one range of values to another in a linear way.
 *
 * @tparam T The type
 * @param value The value to transform
 * @param old_left The left bound of the old interval
 * @param old_right The right bound of the old interval
 * @param new_left The left bound of the new interval
 * @param new_right The right bound of the new interval
 * @return The transformed value
 */
template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T
map(const T& value, const T& old_left, const T& old_right, const T& new_left, const T& new_right)
{
    T m = (new_right - new_left) / (old_right - old_left);
    T b = new_left - m * old_left;

    return m * value + b;
}

namespace detail
{

template<typename glm::length_t N, typename T>
struct dispatch_map
{
    static ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec<N, T> apply(const glm::vec<N, T>& value,
                                                                   const glm::vec<N, T>& old_left,
                                                                   const glm::vec<N, T>& old_right,
                                                                   const glm::vec<N, T>& new_left,
                                                                   const glm::vec<N, T>& new_right)
    {
        return glm::vec<N, T>(0);
    }
};

template<typename T>
struct dispatch_map<1, T>
{
    static ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec<1, T> apply(const glm::vec<1, T>& value,
                                                                   const glm::vec<1, T>& old_left,
                                                                   const glm::vec<1, T>& old_right,
                                                                   const glm::vec<1, T>& new_left,
                                                                   const glm::vec<1, T>& new_right)
    {
        return glm::vec<1, T>(map(value, old_left, old_right, new_left, new_right));
    }
};

template<typename T>
struct dispatch_map<2, T>
{
    static ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec<2, T> apply(const glm::vec<2, T>& value,
                                                                   const glm::vec<2, T>& old_left,
                                                                   const glm::vec<2, T>& old_right,
                                                                   const glm::vec<2, T>& new_left,
                                                                   const glm::vec<2, T>& new_right)
    {
        return glm::vec<2, T>(map(value.x, old_left.x, old_right.x, new_left.x, new_right.x),
                              map(value.y, old_left.y, old_right.y, new_left.y, new_right.y));
    }
};

template<typename T>
struct dispatch_map<3, T>
{
    static ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec<3, T> apply(const glm::vec<3, T>& value,
                                                                   const glm::vec<3, T>& old_left,
                                                                   const glm::vec<3, T>& old_right,
                                                                   const glm::vec<3, T>& new_left,
                                                                   const glm::vec<3, T>& new_right)
    {
        return glm::vec<3, T>(map(value.x, old_left.x, old_right.x, new_left.x, new_right.x),
                              map(value.y, old_left.y, old_right.y, new_left.y, new_right.y),
                              map(value.z, old_left.z, old_right.z, new_left.z, new_right.z));
    }
};

template<typename T>
struct dispatch_map<4, T>
{
    static ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec<4, T> apply(const glm::vec<4, T>& value,
                                                                   const glm::vec<4, T>& old_left,
                                                                   const glm::vec<4, T>& old_right,
                                                                   const glm::vec<4, T>& new_left,
                                                                   const glm::vec<4, T>& new_right)
    {
        return glm::vec<4, T>(map(value.x, old_left.x, old_right.x, new_left.x, new_right.x),
                              map(value.y, old_left.y, old_right.y, new_left.y, new_right.y),
                              map(value.z, old_left.z, old_right.z, new_left.z, new_right.z),
                              map(value.w, old_left.w, old_right.w, new_left.w, new_right.w));
    }
};
}    // namespace detail

/**
 * @brief Map one range of values to another in a linear way.
 *
 * @tparam N The number of vector components
 * @tparam T The type
 * @param value The value to transform
 * @param old_left The left bound of the old interval
 * @param old_right The right bound of the old interval
 * @param new_left The left bound of the new interval
 * @param new_right The right bound of the new interval
 * @return The transformed value
 */
template<glm::length_t N, typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec<N, T> map(const glm::vec<N, T>& value,
                                                      const glm::vec<N, T>& old_left,
                                                      const glm::vec<N, T>& old_right,
                                                      const glm::vec<N, T>& new_left,
                                                      const glm::vec<N, T>& new_right)
{
    return detail::dispatch_map<N, T>::apply(value, old_left, old_right, new_left, new_right);
}

/**
 * @brief Convert from uv [0,1] to ndc [-1,1] space linearly.
 *
 * @tparam T The type
 * @param val The value
 * @return The transformed value
 */
template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T uv2ndc(const T& val)
{
    return T(2) * val - T(1);
}

/**
 * @brief Convert from uv [0,1] to ndc [-1,1] space linearly.
 *
 * @tparam N The number of vector components
 * @tparam T The type
 * @param val The value
 * @return The transformed value
 */
template<glm::length_t N, typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec<N, T> uv2ndc(const glm::vec<N, T>& val)
{
    return glm::vec<N, T>(2) * val - glm::vec<N, T>(1);
}

/**
 * @brief Convert from ndc [-1,1] to uv [0,1] space linearly.
 *
 * @tparam T The type
 * @param val The value
 * @return The transformed value
 */
template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T ndc2uv(const T& val)
{
    return T(0.5) * val + T(0.5);
}

/**
 * @brief Convert from ndc [-1,1] to uv [0,1] space linearly.
 *
 * @tparam N The number of vector components
 * @tparam T The type
 * @param val The value
 * @return The transformed value
 */
template<glm::length_t N, typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec<N, T> ndc2uv(const glm::vec<N, T>& val)
{
    return glm::vec<N, T>(0.5) * val + glm::vec<N, T>(0.5);
}

/**
 * @brief Convert perspective depth values given in ndc space to linear space.
 *
 * @tparam T The type
 * @param ndc_depth The perspective ndc depth
 * @param n The near plane
 * @param f The far plane
 * @return The linear depth
 */
template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T ndc2linearDepth(const T& ndc_depth, const T& n, const T& f)
{
    return (T(2) * n * f) / (f + n - ndc_depth * (f - n));
}

/**
 * @brief Convert linear depth values to perspective ndc depths
 *
 * @tparam T The type
 * @param linear_depth The linear depth
 * @param n The near plane
 * @param f The far plane
 * @return The perspective ndc depth
 */
template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T linearDepth2ndc(const T& linear_depth, const T& n, const T& f)
{
    return (f + n) / (f - n) - (T(2) * f * n / (f - n)) / linear_depth;
}

/**
 * @brief Binary search.
 *
 * @tparam T The type
 * @param sorted_array The sorted array
 * @param value The value to search
 * @param size The length of the array
 *
 * @return The index of the bucket where the item is supposed to go
 */
template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE uint32_t binary_search(const T* sorted_array, T value, uint32_t size)
{
    // Find first element in sorted_array that is larger than value.
    uint32_t left  = 0;
    uint32_t right = size - 1;
    while(left < right)
    {
        uint32_t mid = (left + right) / 2;
        if(sorted_array[mid] < value)
            left = mid + 1;
        else
            right = mid;
    }
    return left;
}

}    // namespace Math
}    // namespace atcg