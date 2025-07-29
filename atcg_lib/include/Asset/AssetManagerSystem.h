#pragma once

#include <Asset/Asset.h>
#include <Core/SystemRegistry.h>

namespace atcg
{
// Based on Hazel Engine (https://github.com/TheCherno/Hazel)
// Modified by Domenic Zingsheim in 2025
using AssetRegistry = std::unordered_map<AssetHandle, AssetMetaData>;
using AssetMap      = std::unordered_map<AssetHandle, atcg::ref_ptr<Asset>>;

/**
 * @brief A class to model an asset manager
 */
class AssetManagerSystem
{
public:
    /**
     * @brief Get an asset.
     * The handle needs to be a valid handle from the registry. If not, nullptr will be returned. If the asset is not
     * already loaded, this will trigger a load.
     *
     * @param handle
     *
     * @return The asset
     */
    atcg::ref_ptr<Asset> getAsset(AssetHandle handle);

    /**
     * @brief Get the metadata of an asset.
     * If the handle is invalid, this will return invalid data.
     *
     * @param handle The handle
     *
     * @return The Meta Data
     */
    const AssetMetaData& getMetaData(AssetHandle handle) const;

    /**
     * @brief Get the Asset registry
     *
     * @return The registry
     */
    const AssetRegistry& getAssetRegistry() const;

    /**
     * @brief Check if the asset handle is valid.
     * This only checks if the handle is registered, not if the Asset is also loaded.
     *
     * @param handle The handle
     *
     * @return True if the handle is valid
     */
    bool isAssetHandleValid(AssetHandle handle) const;

    /**
     * @brief Check if the asset is loaded
     *
     * @param handle The handle
     *
     * @return True if the asset is loaded
     */
    bool isAssetLoaded(AssetHandle handle) const;

    /**
     * @brief Update the name of an asset
     *
     * @param handle The handle
     * @param name The new name
     */
    void updateName(AssetHandle handle, const std::string& name);

    /**
     * @brief Register an asset.
     * This function only puts the metadata into the registry.
     *
     * @param data The data
     *
     * @return The handle to the newly registered asset
     */
    AssetHandle registerAsset(const AssetMetaData& data);

    /**
     * @brief Register an asset.
     * This function only puts the metadata into the registry with the specified handle.
     * Handle is not allowed to already be in-use
     *
     * @param handle The handle
     * @param data The data
     *
     * @return The handle to the newly registered asset- Should be the same as handle
     */
    AssetHandle registerAsset(AssetHandle handle, const AssetMetaData& data);

    /**
     * @brief Register an asset.
     *
     * @param asset The asset to register
     * @param name The name of the asset
     *
     * @return The asset handle (should be the same as asset->handle)
     */
    AssetHandle registerAsset(const atcg::ref_ptr<Asset>& asset, const std::string& name);

    /**
     * @brief Unload an asset.
     * This does not clear the asset from the registry
     *
     * @param handle The handle
     */
    void unloadAsset(AssetHandle handle);

    /**
     * @brief Remove an asset.
     * This completely removes the asset from the manager
     *
     * @param handle The handle
     */
    void removeAsset(AssetHandle handle);

    /**
     * @brief Serialize the asset registry
     *
     * @param registry_path The path
     */
    void serializeRegistry(const std::filesystem::path& registry_path);

    /**
     * @brief Deserialize the registry
     *
     * @param registry_path The path
     */
    void deserializeRegistry(const std::filesystem::path& registry_path);

    /**
     * @brief Serialize the assets
     *
     * @param root_path The root path where all assets will be stored
     */
    void serializeAssets(const std::filesystem::path& root_path);

    /**
     * @brief Clears all assets
     */
    void clear();

    /**
     * @brief Clears and Destroys the asset manager
     */
    void destroy();

protected:
    AssetRegistry _asset_registry;
    AssetMap _loaded_assets;
};

namespace AssetManager
{
/**
 * @brief Get an asset.
 * The handle needs to be a valid handle from the registry. If not, nullptr will be returned. If the asset is not
 * already loaded, this will trigger a load.
 *
 * @param handle
 *
 * @return The asset
 */
ATCG_INLINE atcg::ref_ptr<Asset> getAsset(AssetHandle handle)
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->getAsset(handle);
}

/**
 * @brief Get an asset.
 * The handle needs to be a valid handle from the registry. If not, nullptr will be returned. If the asset is not
 * already loaded, this will trigger a load.
 *
 * @tparam T The asset class
 * @param handle
 *
 * @return The asset
 */
template<typename T>
ATCG_INLINE atcg::ref_ptr<T> getAsset(AssetHandle handle)
{
    auto asset = SystemRegistry::instance()->getSystem<AssetManagerSystem>()->getAsset(handle);
    return std::static_pointer_cast<T>(asset);
}

/**
 * @brief Get the metadata of an asset.
 * If the handle is invalid, this will return invalid data.
 *
 * @param handle The handle
 *
 * @return The Meta Data
 */
ATCG_INLINE const AssetMetaData& getMetaData(AssetHandle handle)
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->getMetaData(handle);
}

/**
 * @brief Get the Asset registry
 *
 * @return The registry
 */
ATCG_INLINE const AssetRegistry& getAssetRegistry()
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->getAssetRegistry();
}

/**
 * @brief Check if the asset handle is valid.
 * This only checks if the handle is registered, not if the Asset is also loaded.
 *
 * @param handle The handle
 *
 * @return True if the handle is valid
 */
ATCG_INLINE bool isAssetHandleValid(AssetHandle handle)
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->isAssetHandleValid(handle);
}

/**
 * @brief Update the name of an asset
 *
 * @param handle The handle
 * @param name The new name
 */
ATCG_INLINE void updateName(AssetHandle handle, const std::string& name)
{
    SystemRegistry::instance()->getSystem<AssetManagerSystem>()->updateName(handle, name);
}

/**
 * @brief Check if the asset is loaded
 *
 * @param handle The handle
 *
 * @return True if the asset is loaded
 */
ATCG_INLINE bool isAssetLoaded(AssetHandle handle)
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->isAssetLoaded(handle);
}

/**
 * @brief Register an asset.
 * This function only puts the metadata into the registry.
 *
 * @param data The data
 *
 * @return The handle to the newly registered asset
 */
ATCG_INLINE AssetHandle registerAsset(const AssetMetaData& data)
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->registerAsset(data);
}

/**
 * @brief Register an asset.
 *
 * @param asset The asset to register
 * @param name The name of the asset
 *
 * @return The asset handle (should be the same as asset->handle)
 */
ATCG_INLINE AssetHandle registerAsset(const atcg::ref_ptr<Asset>& asset, const std::string& name)
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->registerAsset(asset, name);
}

/**
 * @brief Register an asset.
 * This function only puts the metadata into the registry with the specified handle.
 * Handle is not allowed to already be in-use
 *
 * @param handle The handle
 * @param data The data
 *
 * @return The handle to the newly registered asset- Should be the same as handle
 */
ATCG_INLINE AssetHandle registerAsset(AssetHandle handle, const AssetMetaData& data)
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->registerAsset(handle, data);
}

/**
 * @brief Unload an asset.
 * This does not clear the asset from the registry
 *
 * @param handle The handle
 */
ATCG_INLINE void unloadAsset(AssetHandle handle)
{
    SystemRegistry::instance()->getSystem<AssetManagerSystem>()->unloadAsset(handle);
}

/**
 * @brief Remove an asset.
 * This completely removes the asset from the manager
 *
 * @param handle The handle
 */
ATCG_INLINE void removeAsset(AssetHandle handle)
{
    SystemRegistry::instance()->getSystem<AssetManagerSystem>()->removeAsset(handle);
}

/**
 * @brief Serialize the asset registry
 *
 * @param registry_path The path
 */
ATCG_INLINE void serializeRegistry(const std::filesystem::path& registry_path)
{
    SystemRegistry::instance()->getSystem<AssetManagerSystem>()->serializeRegistry(registry_path);
}

/**
 * @brief Deserialize the registry
 *
 * @param registry_path The path
 */
ATCG_INLINE void deserializeRegistry(const std::filesystem::path& registry_path)
{
    SystemRegistry::instance()->getSystem<AssetManagerSystem>()->deserializeRegistry(registry_path);
}

/**
 * @brief Serialize the assets
 *
 * @param root_path The root path where all assets will be stored
 */
ATCG_INLINE void serializeAssets(const std::filesystem::path& root_path)
{
    SystemRegistry::instance()->getSystem<AssetManagerSystem>()->serializeAssets(root_path);
}

/**
 * @brief Clears all assets
 */
ATCG_INLINE void clear()
{
    SystemRegistry::instance()->getSystem<AssetManagerSystem>()->clear();
}
}    // namespace AssetManager

}    // namespace atcg