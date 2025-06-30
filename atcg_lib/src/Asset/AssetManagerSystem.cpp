#include <Asset/AssetManagerSystem.h>

#include <Asset/AssetImporter.h>

namespace atcg
{

namespace detail
{
static std::map<std::filesystem::path, AssetType> s_asset_extension_map = {{".png", AssetType::Texture2D},
                                                                           {".jpg", AssetType::Texture2D},
                                                                           {".jpeg", AssetType::Texture2D}};

static AssetType getAssetTypeFromFileExtension(const std::filesystem::path& extension)
{
    if(s_asset_extension_map.find(extension) == s_asset_extension_map.end())
    {
        return AssetType::None;
    }

    return s_asset_extension_map.at(extension);
}
}    // namespace detail

atcg::ref_ptr<Asset> AssetManagerSystem::getAsset(AssetHandle handle)
{
    if(!isAssetHandleValid(handle))
    {
        return nullptr;
    }

    atcg::ref_ptr<Asset> asset;

    if(isAssetLoaded(handle))
    {
        asset = _loaded_assets[handle];
    }
    else
    {
        const AssetMetaData& meta_data = getMetaData(handle);
        asset                          = AssetImporter::importAsset(meta_data);
        _loaded_assets.insert(std::make_pair(handle, asset));
    }

    return asset;
}

const AssetMetaData& AssetManagerSystem::getMetaData(AssetHandle handle) const
{
    static AssetMetaData s_NullMetadata;
    auto it = _asset_registry.find(handle);
    if(it == _asset_registry.end()) return s_NullMetadata;

    return it->second;
}

const AssetRegistry& AssetManagerSystem::getAssetRegistry() const
{
    return _asset_registry;
}

bool AssetManagerSystem::isAssetHandleValid(AssetHandle handle) const
{
    return handle != 0 && _asset_registry.find(handle) != _asset_registry.end();
}

bool AssetManagerSystem::isAssetLoaded(AssetHandle handle) const
{
    return _loaded_assets.find(handle) != _loaded_assets.end();
}

AssetHandle AssetManagerSystem::registerAsset(const std::filesystem::path& path)
{
    AssetHandle handle;
    AssetMetaData data;
    data.file_path = path;
    data.type      = detail::getAssetTypeFromFileExtension(path.extension());
    _asset_registry.insert(std::make_pair(handle, data));

    return handle;
}

AssetHandle AssetManagerSystem::registerAsset(const atcg::ref_ptr<Asset>& asset, const std::filesystem::path& path)
{
    AssetMetaData data;
    data.file_path = path;
    data.type      = asset->getType();

    _loaded_assets.insert(std::make_pair(asset->handle, asset));

    _asset_registry.insert(std::make_pair(asset->handle, data));

    return asset->handle;
}

AssetHandle AssetManagerSystem::importAsset(const std::filesystem::path& path)
{
    auto handle = registerAsset(path);

    getAsset(handle);

    return handle;
}

}    // namespace atcg