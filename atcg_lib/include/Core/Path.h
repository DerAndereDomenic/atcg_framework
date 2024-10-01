#pragma once

#include <Core/Platform.h>

#include <filesystem>

namespace atcg
{
/**
 * @brief Get the absolute path to the default shader directory
 *
 * @return The path
 */
ATCG_INLINE ATCG_CONST std::filesystem::path shader_directory()
{
    return std::filesystem::path(ATCG_SHADER_DIR);
}

/**
 * @brief Get the absolute path to the default resource directory
 *
 * @return The path
 */
ATCG_INLINE ATCG_CONST std::filesystem::path resource_directory()
{
    return std::filesystem::path(ATCG_RESOURCE_DIR);
}
}    // namespace atcg