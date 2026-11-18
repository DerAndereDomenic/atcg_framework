#pragma once

#include <Core/Platform.h>

#include <cstdlib>
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
    const char* env_dir = std::getenv("ATCG_SHADER_DIR");
    if(env_dir != nullptr && env_dir[0] != '\0')
    {
        return std::filesystem::path(env_dir);
    }
    return std::filesystem::path(ATCG_SHADER_DIR);
}

/**
 * @brief Get the absolute path to the default resource directory
 *
 * @return The path
 */
ATCG_INLINE ATCG_CONST std::filesystem::path resource_directory()
{
    const char* env_dir = std::getenv("ATCG_RESOURCE_DIR");
    if(env_dir != nullptr && env_dir[0] != '\0')
    {
        return std::filesystem::path(env_dir);
    }
    return std::filesystem::path(ATCG_RESOURCE_DIR);
}

/**
 * @brief Get the absolute path to the directory that holds the compiled PTX modules.
 *
 * If the environment variable ATCG_PTX_DIR is set, its value is returned. Otherwise, the build-time output directory
 * of the PTX modules is returned.
 *
 * @return The path
 */
ATCG_INLINE std::filesystem::path ptx_directory()
{
    const char* env_dir = std::getenv("ATCG_PTX_DIR");
    if(env_dir != nullptr && env_dir[0] != '\0')
    {
        return std::filesystem::path(env_dir);
    }
    return std::filesystem::path(ATCG_PTX_DIR);
}
}    // namespace atcg