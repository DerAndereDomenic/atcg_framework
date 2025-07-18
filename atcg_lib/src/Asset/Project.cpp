#include <Asset/Project.h>
#include <Asset/AssetManagerSystem.h>

namespace atcg
{
atcg::ref_ptr<Project> Project::create(const std::filesystem::path& path)
{
    s_active_project                        = atcg::make_ref<Project>();
    s_active_project->_project_path         = path;
    s_active_project->_asset_pack_directory = path / "AssetPack.json";

    return s_active_project;
}

atcg::ref_ptr<Project> Project::load(const std::filesystem::path& path)
{
    s_active_project                = atcg::make_ref<Project>();
    s_active_project->_project_path = path.parent_path();

    // 1. Load project information
    s_active_project->deserializeProjectInformation();

    // 2. Load Asset registry
    AssetManager::clear();    // Clear current assets
    AssetManager::deserializeRegistry(s_active_project->_asset_pack_directory);

    //? 3. Import assets (this is done lazily?)

    return s_active_project;
}

void Project::save()
{
    // 1. Create necessary directories
    std::filesystem::create_directories(_project_path / "assets" / "graphs");
    std::filesystem::create_directories(_project_path / "assets" / "materials");
    std::filesystem::create_directories(_project_path / "assets" / "textures");
    std::filesystem::create_directories(_project_path / "assets" / "scenes");
    std::filesystem::create_directories(_project_path / "assets" / "scripts");
    std::filesystem::create_directories(_project_path / "assets" / "shader");

    // 2. Serialize Asset Registry
    AssetManager::serializeRegistry(_asset_pack_directory);

    // 3. Serialize Assets
    AssetManager::serializeAssets(_project_path / "assets");

    // 4. Serialize Project information
    serializeProjectInformation();
}

void Project::setActiveScene(const atcg::ref_ptr<Scene>& scene)
{
    AssetHandle handle = scene->handle;

    if(!AssetManager::isAssetHandleValid(handle))
    {
        AssetManager::registerAsset(scene, "scene");
    }

    setActiveScene(handle);
}

void Project::setActiveScene(AssetHandle handle)
{
    _active_scene = handle;
}

atcg::ref_ptr<Scene> Project::getActiveScene() const
{
    return AssetManager::getAsset<Scene>(_active_scene);
}

const atcg::ref_ptr<Project>& Project::getActive()
{
    return s_active_project;
}

void Project::saveActive()
{
    if(s_active_project) s_active_project->save();
}

void Project::serializeProjectInformation_ver1()
{
    nlohmann::json j;

    j["Version"] = "1.0";

    j["Assets"] = _asset_pack_directory.filename();
    j["Scene"]  = (uint64_t)_active_scene;

    std::ofstream o(_project_path / "Project.json");
    o << std::setw(4) << j << std::endl;
}

void Project::serializeProjectInformation()
{
    serializeProjectInformation_ver1();
}

void Project::deserializeProjectInformation_ver1(const nlohmann::json& j)
{
    auto asset_path       = j["Assets"];
    _asset_pack_directory = _project_path / asset_path;
    _active_scene         = (AssetHandle)j["Scene"];
}

void Project::deserializeProjectInformation()
{
    std::ifstream i(_project_path / "Project.json");
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
        deserializeProjectInformation_ver1(j);
    }
}
}    // namespace atcg