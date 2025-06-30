#pragma once

#include <Core/UUID.h>

#include <filesystem>

namespace atcg
{

// Based on Hazel Engine (https://github.com/TheCherno/Hazel)
// Modified by Domenic Zingsheim in 2025
using AssetHandle = UUID;

enum class AssetType : uint16_t
{
    None = 0,
    // Scene, TODO
    Texture2D,
    Material,
    // Shader, TODO
    // Graph TODO
};

class Asset
{
public:
    AssetHandle handle;

    virtual AssetType getType() const = 0;
};

struct AssetMetaData
{
    AssetType type = AssetType::None;
    std::filesystem::path file_path;

    operator bool() const { return type != AssetType::None; }
};
}    // namespace atcg