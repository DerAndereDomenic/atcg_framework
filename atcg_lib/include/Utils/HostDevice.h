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

/**
 * @brief Create a diagonal matrix from a vector
 * @param v The input vector
 *
 * @return The diagonal matrix
 */
ATCG_INLINE ATCG_DEVICE glm::mat3 diag(const glm::vec3& v)
{
    glm::mat3 M = glm::mat3(0);

    M[0][0] = v.x;
    M[1][1] = v.y;
    M[2][2] = v.z;

    return M;
}
}    // namespace atcg