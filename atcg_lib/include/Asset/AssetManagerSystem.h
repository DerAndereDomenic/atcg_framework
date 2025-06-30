#pragma once

#include <Asset/Asset.h>
#include <Core/SystemRegistry.h>

namespace atcg
{
// Based on Hazel Engine (https://github.com/TheCherno/Hazel)
// Modified by Domenic Zingsheim in 2025
using AssetRegistry = std::unordered_map<AssetHandle, AssetMetaData>;
using AssetMap      = std::unordered_map<AssetHandle, atcg::ref_ptr<Asset>>;

class AssetManagerSystem
{
public:
    atcg::ref_ptr<Asset> getAsset(AssetHandle handle);

    const AssetMetaData& getMetaData(AssetHandle handle) const;

    const AssetRegistry& getAssetRegistry() const;

    bool isAssetHandleValid(AssetHandle handle) const;

    bool isAssetLoaded(AssetHandle handle) const;

    AssetHandle registerAsset(const std::filesystem::path& path);

    AssetHandle registerAsset(const atcg::ref_ptr<Asset>& asset, const std::filesystem::path& path);

    AssetHandle importAsset(const std::filesystem::path& path);

protected:
    AssetRegistry _asset_registry;
    AssetMap _loaded_assets;
};

namespace AssetManager
{
ATCG_INLINE atcg::ref_ptr<Asset> getAsset(AssetHandle handle)
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->getAsset(handle);
}

template<typename T>
ATCG_INLINE atcg::ref_ptr<T> getAsset(AssetHandle handle)
{
    auto asset = SystemRegistry::instance()->getSystem<AssetManagerSystem>()->getAsset(handle);
    return std::static_pointer_cast<T>(asset);
}

ATCG_INLINE const AssetMetaData& getMetaData(AssetHandle handle)
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->getMetaData(handle);
}

ATCG_INLINE const AssetRegistry& getAssetRegistry()
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->getAssetRegistry();
}

ATCG_INLINE bool isAssetHandleValid(AssetHandle handle)
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->isAssetHandleValid(handle);
}

ATCG_INLINE bool isAssetLoaded(AssetHandle handle)
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->isAssetLoaded(handle);
}

ATCG_INLINE AssetHandle registerAsset(const std::filesystem::path& path)
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->registerAsset(path);
}

ATCG_INLINE AssetHandle registerAsset(const atcg::ref_ptr<Asset>& asset, const std::filesystem::path& path)
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->registerAsset(asset, path);
}

ATCG_INLINE AssetHandle importAsset(const std::filesystem::path& path)
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->importAsset(path);
}
}    // namespace AssetManager

}    // namespace atcg