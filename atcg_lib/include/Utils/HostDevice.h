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
}    // namespace atcg