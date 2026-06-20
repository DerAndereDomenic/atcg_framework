#include <Asset/AssetManagerSystem.h>

#include <Core/Path.h>
#include <Asset/AssetImporter.h>
#include <Asset/AssetExporter.h>
#include <Asset/Project.h>
#include <DataStructure/GraphLoader.h>

#include <json.hpp>

namespace atcg
{

namespace detail
{
static std::map<std::filesystem::path, AssetType> s_asset_extension_map = {{".png", AssetType::Texture2D},
                                                                           {".jpg", AssetType::Texture2D},
                                                                           {".jpeg", AssetType::Texture2D},
                                                                           {".graph", AssetType::Graph},
                                                                           {".mat", AssetType::Material},
                                                                           {".scene", AssetType::Scene}};

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
        asset = AssetImporter::importAsset(Project::getActive()->getFilePath() / "assets", handle, meta_data);
        if(asset) _loaded_assets.insert(std::make_pair(handle, asset));
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

AssetHandle AssetManagerSystem::registerAsset(const AssetMetaData& data)
{
    AssetHandle handle;
    return registerAsset(handle, data);
}

AssetHandle AssetManagerSystem::registerAsset(AssetHandle handle, const AssetMetaData& data)
{
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

namespace detail
{
ATCG_INLINE void deserialize_registry_ver1(AssetRegistry& registry, const nlohmann::json& j)
{
    auto json_registry = j["Registry"];

    for(auto entry: json_registry)
    {
        AssetHandle handle = (AssetHandle)entry["Handle"];
        std::string name   = entry["Name"];
        AssetType type     = stringToAssetType(std::string(entry["Type"]));

        AssetMetaData data;
        data.name        = name;
        data.type        = type;
        registry[handle] = data;
    }
}
}    // namespace detail

void AssetManagerSystem::deserializeRegistry(const std::filesystem::path& registry_path)
{
    std::ifstream i(registry_path);
    nlohmann::json j;
    i >> j;

    if(!j.contains("Version"))
    {
        ATCG_WARN("Got invalid Project file with unrecognized version. Abort...");
        return;
    }

    std::string version = j["Version"];

    if(version == "1.0")
    {
        detail::deserialize_registry_ver1(_asset_registry, j);
    }
}

void AssetManagerSystem::serializeAssets(const std::filesystem::path& root_path)
{
    for(auto entry: _asset_registry)
    {
        AssetExporter::exportAsset(root_path, getAsset(entry.first), getMetaData(entry.first));
    }
}

void AssetManagerSystem::clear()
{
    _asset_registry.clear();
    _loaded_assets.clear();
}

void AssetManagerSystem::destroy()
{
    clear();
}

void AssetManagerSystem::loadStandardAssets()
{
    _sphere_mesh   = atcg::IO::read_mesh((atcg::resource_directory() / "sphere_low.obj").string());
    _cylinder_mesh = atcg::IO::read_mesh((atcg::resource_directory() / "cylinder.obj").string());

    auto img = IO::imread((atcg::resource_directory() / "LUT.hdr").string());
    TextureSpecification spec_lut;
    spec_lut.width             = img->width();
    spec_lut.height            = img->height();
    spec_lut.format            = TextureFormat::RGBFLOAT;
    spec_lut.sampler.wrap_mode = TextureWrapMode::CLAMP_TO_EDGE;
    _lut_texture               = atcg::Texture2D::create(img, spec_lut);

    {
        glm::vec3 eye = glm::vec3(0);

        std::vector<atcg::Vertex> points;
        points.push_back(atcg::Vertex(eye, glm::vec3(1)));
        points.push_back(atcg::Vertex(eye + glm::vec3(-0.5, -0.5, 1.0f), glm::vec3(1)));
        points.push_back(atcg::Vertex(eye + glm::vec3(0.5, -0.5, 1.0f), glm::vec3(1)));
        points.push_back(atcg::Vertex(eye + glm::vec3(0.5, 0.5, 1.0f), glm::vec3(1)));
        points.push_back(atcg::Vertex(eye + glm::vec3(-0.5, 0.5, 1.0f), glm::vec3(1)));

        std::vector<atcg::Edge> edges;
        edges.push_back({glm::vec2(0, 1), glm::vec3(1), 0.01f});
        edges.push_back({glm::vec2(0, 2), glm::vec3(1), 0.01f});
        edges.push_back({glm::vec2(0, 3), glm::vec3(1), 0.01f});
        edges.push_back({glm::vec2(0, 4), glm::vec3(1), 0.01f});

        edges.push_back({glm::vec2(1, 2), glm::vec3(1), 0.01f});
        edges.push_back({glm::vec2(2, 3), glm::vec3(1), 0.01f});
        edges.push_back({glm::vec2(3, 4), glm::vec3(1), 0.01f});
        edges.push_back({glm::vec2(4, 1), glm::vec3(1), 0.01f});

        _camera_frustum = atcg::Graph::createGraph(points, edges);
    }

    {
        std::vector<atcg::Vertex> vertices = {
            atcg::Vertex(glm::vec3(-1, -1, 0),
                         glm::vec3(1),
                         glm::vec3(0, 0, 1),
                         glm::vec3(1, 0, 0),
                         glm::vec3(0, 0, 0)),
            atcg::Vertex(glm::vec3(1, -1, 0), glm::vec3(1), glm::vec3(0, 0, 1), glm::vec3(1, 0, 0), glm::vec3(1, 0, 0)),
            atcg::Vertex(glm::vec3(1, 1, 0), glm::vec3(1), glm::vec3(0, 0, 1), glm::vec3(1, 0, 0), glm::vec3(1, 1, 0)),
            atcg::Vertex(glm::vec3(-1, 1, 0),
                         glm::vec3(1),
                         glm::vec3(0, 0, 1),
                         glm::vec3(1, 0, 0),
                         glm::vec3(0, 1, 0))};

        std::vector<glm::u32vec3> edges = {glm::u32vec3(0, 1, 2), glm::u32vec3(0, 2, 3)};

        _quad = atcg::Graph::createTriangleMesh(vertices, edges);
    }

    {
        std::vector<atcg::Vertex> points;
        points.push_back(atcg::Vertex(glm::vec3(0.5f, 0.5f, -0.5f), glm::vec3(1)));
        points.push_back(atcg::Vertex(glm::vec3(0.5f, -0.5f, -0.5f), glm::vec3(1)));
        points.push_back(atcg::Vertex(glm::vec3(0.5f, 0.5f, 0.5f), glm::vec3(1)));
        points.push_back(atcg::Vertex(glm::vec3(0.5f, -0.5f, 0.5f), glm::vec3(1)));
        points.push_back(atcg::Vertex(glm::vec3(-0.5f, 0.5f, -0.5f), glm::vec3(1)));
        points.push_back(atcg::Vertex(glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(1)));
        points.push_back(atcg::Vertex(glm::vec3(-0.5f, 0.5f, 0.5f), glm::vec3(1)));
        points.push_back(atcg::Vertex(glm::vec3(-0.5f, -0.5f, 0.5f), glm::vec3(1)));

        std::vector<glm::u32vec3> faces;
        faces.push_back(glm::u32vec3(4, 2, 0));
        faces.push_back(glm::u32vec3(2, 7, 3));
        faces.push_back(glm::u32vec3(6, 5, 7));
        faces.push_back(glm::u32vec3(1, 7, 5));
        faces.push_back(glm::u32vec3(0, 3, 1));
        faces.push_back(glm::u32vec3(4, 1, 5));
        faces.push_back(glm::u32vec3(4, 6, 2));
        faces.push_back(glm::u32vec3(2, 6, 7));
        faces.push_back(glm::u32vec3(6, 4, 5));
        faces.push_back(glm::u32vec3(1, 3, 7));
        faces.push_back(glm::u32vec3(0, 2, 3));
        faces.push_back(glm::u32vec3(4, 0, 1));

        _cube_mesh = atcg::Graph::createTriangleMesh(points, faces);
    }

    _dummy_skybox = atcg::make_ref<Skybox>();
}

}    // namespace atcg