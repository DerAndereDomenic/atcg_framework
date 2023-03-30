#pragma once

#include <memory>

namespace atcg
{

/**
 * @brief These memory objects currently only replace std::shared_ptr and std::unique_ptr.
 * Later, we plan to use this interface to also handle cuda (device) memory.
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
}    // namespace atcg