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
    Scene,
    Texture2D,
    Material,
    Graph,
    Script,
    Shader
};

class Asset
{
public:
    AssetHandle handle;

    virtual AssetType getType() const = 0;
};

struct AssetMetaData
{
    AssetType type   = AssetType::None;
    std::string name = "Asset";

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
        case AssetType::Scene:
            return "AssetType::Scene";
        case AssetType::Script:
            return "AssetType::Script";
        case AssetType::Shader:
            return "AssetType::Shader";
    }

    return "AssetType::<Invalid>";
}

ATCG_INLINE AssetType stringToAssetType(std::string_view str)
{
    if(str == "AssetType::None") return AssetType::None;
    if(str == "AssetType::Material") return AssetType::Material;
    if(str == "AssetType::Texture2D") return AssetType::Texture2D;
    if(str == "AssetType::Graph") return AssetType::Graph;
    if(str == "AssetType::Scene") return AssetType::Scene;
    if(str == "AssetType::Script") return AssetType::Script;
    if(str == "AssetType::Shader") return AssetType::Shader;

    // Unknown string:
    return AssetType::None;    // or AssetType::<Invalid> if you have it
}
}    // namespace atcg