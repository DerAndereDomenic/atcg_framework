#pragma once

#include <Asset/Asset.h>
#include <Core/SystemRegistry.h>
#include <DataStructure/Graph.h>
#include <Renderer/Texture.h>
#include <DataStructure/Skybox.h>
#include <Renderer/Material.h>

namespace atcg
{
// Based on Hazel Engine (https://github.com/TheCherno/Hazel)
// Modified by Domenic Zingsheim in 2025
using AssetRegistry = std::unordered_map<AssetHandle, AssetMetaData>;
using AssetMap      = std::unordered_map<AssetHandle, atcg::ref_ptr<Asset>>;

/**
 * @brief A class to model an asset manager
 */
class ATCG_API AssetManagerSystem
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
     * @param show_in_editor Whether the asset should be shown in the editor
     * @param serialize Whether the asset should be serialized
     *
     * @return The asset handle (should be the same as asset->handle)
     */
    AssetHandle registerAsset(const atcg::ref_ptr<Asset>& asset,
                              const std::string& name,
                              bool show_in_editor = true,
                              bool serialize      = true);

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
     * @note This does not clear the default assets
     */
    void clear();

    /**
     * @brief Clears and Destroys the asset manager
     */
    void destroy();

    /**
     * @brief Load standard assets like sphere mesh, cylinder mesh, lut texture
     */
    void loadStandardAssets();

    /**
     * @brief Register standard assets
     */
    void registerStandardAssets();

    /**
     * @brief Get the standard sphere mesh
     *
     * @return The sphere mesh
     */
    ATCG_INLINE atcg::ref_ptr<Graph> getSphereMesh() const { return _sphere_mesh; }

    /**
     * @brief Get the standard cylinder mesh
     *
     * @return The cylinder mesh
     */
    ATCG_INLINE atcg::ref_ptr<Graph> getCylinderMesh() const { return _cylinder_mesh; }

    /**
     * @brief Get the standard lut texture
     *
     * @return The lut texture
     */
    ATCG_INLINE atcg::ref_ptr<Texture2D> getLUTTexture() const { return _lut_texture; }

    /**
     * @brief Get the camera frustum mesh
     *
     * @return The camera frustum mesh
     */
    ATCG_INLINE atcg::ref_ptr<Graph> getCameraFrustumMesh() const { return _camera_frustum; }

    /**
     * @brief Get the quad mesh
     *
     * @return The quad mesh
     */
    ATCG_INLINE atcg::ref_ptr<Graph> getQuadMesh() const { return _quad; }

    /**
     * @brief Get the cube mesh
     *
     * @return The cube mesh
     */
    ATCG_INLINE atcg::ref_ptr<Graph> getCubeMesh() const { return _cube_mesh; }

    /**
     * @brief Get the dummy skybox
     *
     * @return The dummy skybox
     */
    ATCG_INLINE atcg::ref_ptr<Skybox> getDummySkybox() const { return _dummy_skybox; }

    /**
     * @brief Get the default material
     *
     * @return The default material
     */
    ATCG_INLINE atcg::ref_ptr<Material> getDefaultMaterial() const { return _default_material; }

protected:
    AssetRegistry _asset_registry;
    AssetMap _loaded_assets;

private:
    atcg::ref_ptr<Graph> _sphere_mesh;
    atcg::ref_ptr<Graph> _cylinder_mesh;
    atcg::ref_ptr<Texture2D> _lut_texture;
    atcg::ref_ptr<Graph> _camera_frustum;
    atcg::ref_ptr<Graph> _quad;
    atcg::ref_ptr<Graph> _cube_mesh;
    atcg::ref_ptr<Skybox> _dummy_skybox;
    atcg::ref_ptr<Material> _default_material;
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
 * @param show_in_editor Whether the asset should be shown in the editor
 * @param serialize Whether the asset should be serialized
 *
 * @return The asset handle (should be the same as asset->handle)
 */
ATCG_INLINE AssetHandle registerAsset(const atcg::ref_ptr<Asset>& asset,
                                      const std::string& name,
                                      bool show_in_editor = true,
                                      bool serialize      = true)
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->registerAsset(asset,
                                                                                      name,
                                                                                      show_in_editor,
                                                                                      serialize);
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
 * @note This does not clear the default assets
 */
ATCG_INLINE void clear()
{
    SystemRegistry::instance()->getSystem<AssetManagerSystem>()->clear();
}

/**
 * @brief Load standard assets like sphere mesh, cylinder mesh, lut texture
 */
ATCG_INLINE void loadStandardAssets()
{
    SystemRegistry::instance()->getSystem<AssetManagerSystem>()->loadStandardAssets();
}

/**
 * @brief Get the standard sphere mesh
 *
 * @return The sphere mesh
 */
ATCG_INLINE atcg::ref_ptr<Graph> getSphereMesh()
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->getSphereMesh();
}

/**
 * @brief Get the standard cylinder mesh
 *
 * @return The cylinder mesh
 */
ATCG_INLINE atcg::ref_ptr<Graph> getCylinderMesh()
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->getCylinderMesh();
}

/**
 * @brief Get the standard lut texture
 *
 * @return The lut texture
 */
ATCG_INLINE atcg::ref_ptr<Texture2D> getLUTTexture()
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->getLUTTexture();
}

/**
 * @brief Get the camera frustum mesh
 *
 * @return The camera frustum mesh
 */
ATCG_INLINE atcg::ref_ptr<Graph> getCameraFrustumMesh()
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->getCameraFrustumMesh();
}

/**
 * @brief Get the quad mesh
 *
 * @return The quad mesh
 */
ATCG_INLINE atcg::ref_ptr<Graph> getQuadMesh()
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->getQuadMesh();
}

/**
 * @brief Get the cube mesh
 *
 * @return The cube mesh
 */
ATCG_INLINE atcg::ref_ptr<Graph> getCubeMesh()
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->getCubeMesh();
}

/**
 * @brief Get the dummy skybox
 *
 * @return The dummy skybox
 */
ATCG_INLINE atcg::ref_ptr<Skybox> getDummySkybox()
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->getDummySkybox();
}

/**
 * @brief Get the default material
 *
 * @return The default material
 */
ATCG_INLINE atcg::ref_ptr<Material> getDefaultMaterial()
{
    return SystemRegistry::instance()->getSystem<AssetManagerSystem>()->getDefaultMaterial();
}

}    // namespace AssetManager

}    // namespace atcg