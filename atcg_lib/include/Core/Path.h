#pragma once

#include <Core/Platform.h>

#include <filesystem>

namespace atcg
{
ATCG_INLINE ATCG_CONST std::filesystem::path shader_directory()
{
    return std::filesystem::path(ATCG_SHADER_DIR);
}

ATCG_INLINE ATCG_CONST std::filesystem::path resource_directory()
{
    return std::filesystem::path(ATCG_RESOURCE_DIR);
}
}    // namespace atcg