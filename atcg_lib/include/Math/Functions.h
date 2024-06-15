#pragma once

#include <Core/CUDA.h>
#include <Core/glm.h>

namespace atcg
{
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
map(const T& value, const T& old_left, const T& old_right, const T& new_left, const T& new_right);

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
                                                      const glm::vec<N, T>& new_right);

/**
 * @brief Convert from uv [0,1] to ndc [-1,1] space linearly.
 *
 * @tparam T The type
 * @param val The value
 * @return The transformed value
 */
template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T uv2ndc(const T& val);

/**
 * @brief Convert from uv [0,1] to ndc [-1,1] space linearly.
 *
 * @tparam N The number of vector components
 * @tparam T The type
 * @param val The value
 * @return The transformed value
 */
template<glm::length_t N, typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec<N, T> uv2ndc(const glm::vec<N, T>& val);

/**
 * @brief Convert from ndc [-1,1] to uv [0,1] space linearly.
 *
 * @tparam T The type
 * @param val The value
 * @return The transformed value
 */
template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T ndc2uv(const T& val);

/**
 * @brief Convert from ndc [-1,1] to uv [0,1] space linearly.
 *
 * @tparam N The number of vector components
 * @tparam T The type
 * @param val The value
 * @return The transformed value
 */
template<glm::length_t N, typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec<N, T> ndc2uv(const glm::vec<N, T>& val);

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
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T ndc2linearDepth(const T& ndc_depth, const T& n, const T& f);

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
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T linearDepth2ndc(const T& linear_depth, const T& n, const T& f);

}    // namespace Math

}    // namespace atcg

#include "../../src/Math/FunctionsDetail.h"