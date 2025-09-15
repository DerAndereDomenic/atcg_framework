#pragma once

#include <Core/UUID.h>

#include <Core/Platform.h>

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
    Texture3D,
    Material,
    Graph,
    Script,
    Shader
};

/**
 * @brief A class to model an asset
 */
class Asset
{
public:
    AssetHandle handle;

    /**
     * @brief Get the type of the asset
     *
     * @return The asset type
     */
    virtual AssetType getType() const = 0;
};

/**
 * @brief A class to model asset meta data
 */
struct AssetMetaData
{
    AssetType type   = AssetType::None;
    std::string name = "Asset";

    operator bool() const { return type != AssetType::None; }
};

/**
 * @brief Convert Asset Type to string
 *
 * @param type The type
 *
 * @return The type as string
 */
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
        case AssetType::Texture3D:
            return "AssetType::Texture3D";
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

/**
 * @brief Convert a string to the asset type
 *
 * @param str The string
 *
 * @return The type
 */
ATCG_INLINE AssetType stringToAssetType(std::string_view str)
{
    if(str == "AssetType::None") return AssetType::None;
    if(str == "AssetType::Material") return AssetType::Material;
    if(str == "AssetType::Texture2D") return AssetType::Texture2D;
    if(str == "AssetType::Texture3D") return AssetType::Texture3D;
    if(str == "AssetType::Graph") return AssetType::Graph;
    if(str == "AssetType::Scene") return AssetType::Scene;
    if(str == "AssetType::Script") return AssetType::Script;
    if(str == "AssetType::Shader") return AssetType::Shader;

    // Unknown string:
    return AssetType::None;    // or AssetType::<Invalid> if you have it
}
}    // namespace atcg