#pragma once

#include <Core/API.h>

#include <filesystem>

namespace atcg
{
/**
 * @brief Get the directory that contains the atcg_lib shared library (atcg_lib.dll / libatcg_lib.so).
 * Plugins are searched in this directory first, so that each build of the library loads the plugins built against it.
 *
 * @return The path
 */
ATCG_API std::filesystem::path library_directory();

/**
 * @brief Get the path of the executable of the current process
 *
 * @return The path
 */
ATCG_API std::filesystem::path executable_path();

/**
 * @brief Get the per-user configuration directory of the framework.
 * This is %APPDATA%/atcg on Windows and $XDG_CONFIG_HOME/atcg (or ~/.config/atcg) on Linux. The directory is created if
 * it does not exist yet.
 *
 * @return The path
 */
ATCG_API std::filesystem::path user_config_directory();

/**
 * @brief Get the absolute path to the default shader directory.
 * This is the shader directory of the source tree, unless the environment variable ATCG_SHADER_DIR is set.
 *
 * @return The path
 */
ATCG_API std::filesystem::path shader_directory();

/**
 * @brief Get the absolute path to the default resource directory.
 * This is the resource directory of the source tree, unless the environment variable ATCG_RESOURCE_DIR is set.
 *
 * @return The path
 */
ATCG_API std::filesystem::path resource_directory();

/**
 * @brief Get the absolute path to the directory that holds the compiled PTX modules.
 * This is the build-time output directory of the PTX modules, unless the environment variable ATCG_PTX_DIR is set.
 *
 * @return The path
 */
ATCG_API std::filesystem::path ptx_directory();
}    // namespace atcg
