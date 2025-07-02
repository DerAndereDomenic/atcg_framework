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
    Graph
    // Shader, TODO
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

ATCG_INLINE const char* assetTypeToString(AssetType type)
{
    switch(type)
    {
        case AssetType::None:
            return "AssetType::None";
        case AssetType::Material:
            return "AssetType::Material";
        case AssetType::Texture2D:
            return "AssetType::Texture2D";
        case AssetType::Graph:
            return "AssetType::Graph";
    }

    return "AssetType::<Invalid>";
}
}    // namespace atcg