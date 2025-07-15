#include <Asset/AssetManagerSystem.h>

#include <Asset/AssetImporter.h>
#include <Asset/AssetExporter.h>

#include <json.hpp>

namespace atcg
{

namespace detail
{
static std::map<std::filesystem::path, AssetType> s_asset_extension_map = {{".png", AssetType::Texture2D},
                                                                           {".jpg", AssetType::Texture2D},
                                                                           {".jpeg", AssetType::Texture2D},
                                                                           {".graph", AssetType::Graph},
                                                                           {".mat", AssetType::Material}};

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

void AssetManagerSystem::updateName(AssetHandle handle, const std::string& name)
{
    if(!isAssetHandleValid(handle)) return;
    auto& data = _asset_registry[handle];
    data.name  = name;
}

AssetHandle AssetManagerSystem::registerAsset(const std::filesystem::path& path)
{
    AssetHandle handle;
    AssetMetaData data;
    data.type               = detail::getAssetTypeFromFileExtension(path.extension());
    data.name               = path.stem().string();
    _asset_registry[handle] = data;

    return handle;
}

AssetHandle AssetManagerSystem::registerAsset(const AssetMetaData& data)
{
    AssetHandle handle;
    _asset_registry[handle] = data;

    return handle;
}

AssetHandle AssetManagerSystem::registerAsset(const atcg::ref_ptr<Asset>& asset, const std::string& name)
{
    AssetMetaData data;
    data.type = asset->getType();
    data.name = name;

    _loaded_assets[asset->handle] = asset;

    _asset_registry[asset->handle] = data;

    return asset->handle;
}

AssetHandle AssetManagerSystem::registerAsset(const atcg::ref_ptr<Asset>& asset, const AssetMetaData& data)
{
    _loaded_assets[asset->handle] = asset;

    _asset_registry[asset->handle] = data;

    return asset->handle;
}

AssetHandle AssetManagerSystem::importAsset(const std::filesystem::path& path)
{
    auto handle = registerAsset(path);

    getAsset(handle);

    return handle;
}

void AssetManagerSystem::unloadAsset(AssetHandle handle)
{
    if(!isAssetHandleValid(handle)) return;

    if(isAssetLoaded(handle))
    {
        _loaded_assets.erase(handle);
    }
}

void AssetManagerSystem::removeAsset(AssetHandle handle)
{
    if(!isAssetHandleValid(handle)) return;

    if(isAssetLoaded(handle))
    {
        unloadAsset(handle);
    }

    // Remove from registry
    _asset_registry.erase(handle);
}

namespace detail
{
ATCG_INLINE void serialize_registry_ver1(const AssetRegistry& registry, const std::filesystem::path& registry_path)
{
    nlohmann::json j;

    j["Version"] = "1.0";

    std::vector<nlohmann::json> serialized_registry;

    for(auto entry: registry)
    {
        AssetHandle handle = entry.first;
        AssetMetaData data = entry.second;

        nlohmann::json asset_entry;
        asset_entry["Handle"] = (uint64_t)handle;
        asset_entry["Type"]   = assetTypeToString(data.type);
        asset_entry["Name"]   = data.name;

        serialized_registry.push_back(asset_entry);
    }

    j["Registry"] = serialized_registry;

    std::ofstream o(registry_path);
    o << std::setw(4) << j << std::endl;
}
}    // namespace detail

void AssetManagerSystem::serializeRegistry(const std::filesystem::path& registry_path)
{
    detail::serialize_registry_ver1(_asset_registry, registry_path);
}

void AssetManagerSystem::serializeAssets(const std::filesystem::path& root_path)
{
    for(auto entry: _asset_registry)
    {
        AssetExporter::exportAsset(root_path, getAsset(entry.first), getMetaData(entry.first));
    }
}

void AssetManagerSystem::destroy()
{
    _asset_registry.clear();
    _loaded_assets.clear();
}

}    // namespace atcg