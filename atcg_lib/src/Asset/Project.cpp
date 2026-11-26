#include <Asset/Project.h>
#include <Asset/AssetManagerSystem.h>
#include <Scene/RevisionStack.h>
#include <Plugin/Plugin.h>

#ifdef ATCG_CUDA_BACKEND
    #include <Integrator/IntegratorRegistry.h>
#endif

namespace atcg
{
atcg::ref_ptr<Project> Project::create(const std::filesystem::path& path)
{
    s_active_project                        = atcg::make_ref<Project>();
    s_active_project->_project_path         = path;
    s_active_project->_asset_pack_directory = path / "AssetPack.json";

#ifdef ATCG_CUDA_BACKEND
    s_active_project->_integrator_settings = atcg::IntegratorRegistry::createIntegratorSettings("VolPathtracing");
#endif

    return s_active_project;
}

atcg::ref_ptr<Project> Project::load(const std::filesystem::path& path)
{
    RevisionStack::clearChache();
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

void Project::save(const std::filesystem::path& path)
{
    _project_path         = path;
    _asset_pack_directory = path / "AssetPack.json";

    save();
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

void Project::saveActive(const std::filesystem::path& path)
{
    if(s_active_project) s_active_project->save(path);
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

void Project::serializeIntegratorSettings_ver1()
{
#ifdef ATCG_CUDA_BACKEND
    if(_integrator_settings)
    {
        nlohmann::json j = _integrator_settings->toJson();

        j["Type"] = _integrator_settings->getType();
        std::ofstream o(_project_path / "IntegratorSettings.json");
        o << std::setw(4) << j << std::endl;
    }
#endif
}

void Project::serializeProjectInformation()
{
    serializeProjectInformation_ver1();
    serializeIntegratorSettings_ver1();
    serializePlugins_ver1();
}

void Project::deserializeProjectInformation_ver1(const nlohmann::json& j)
{
    auto asset_path       = j["Assets"];
    _asset_pack_directory = _project_path / asset_path;
    _active_scene         = (AssetHandle)j["Scene"];
}

void Project::deserializeIntegratorSettings_ver1(const nlohmann::json& j)
{
#ifdef ATCG_CUDA_BACKEND
    if(j.contains("Type"))
    {
        std::string type = j["Type"];

        _integrator_settings = atcg::IntegratorRegistry::createIntegratorSettings(type);

        if(_integrator_settings)
        {
            _integrator_settings->fromJson(j);
        }
    }
#endif
}

void Project::deserializeProjectInformation()
{
    std::ifstream project_info_file(_project_path / "Project.json");
    nlohmann::json project_info_json;
    project_info_file >> project_info_json;

    if(!project_info_json.contains("Version"))
    {
        ATCG_WARN("Got invalid Project file with unrecognized version. Abort...");
        return;
    }

    std::string version = project_info_json["Version"];

    if(version == "1.0")
    {
        deserializeProjectInformation_ver1(project_info_json);

        if(std::filesystem::exists(_project_path / "IntegratorSettings.json"))
        {
            std::ifstream integrator_settings_file(_project_path / "IntegratorSettings.json");
            nlohmann::json integrator_settings_json;
            integrator_settings_file >> integrator_settings_json;

            deserializeIntegratorSettings_ver1(integrator_settings_json);
        }
        else
        {
// Load default settings
#ifdef ATCG_CUDA_BACKEND
            _integrator_settings = atcg::IntegratorRegistry::createIntegratorSettings("VolPathtracing");
#endif
        }

        if(std::filesystem::exists(_project_path / "Plugins.json"))
        {
            std::ifstream plugins_file(_project_path / "Plugins.json");
            nlohmann::json plugins_json;
            plugins_file >> plugins_json;

            deserializePlugins_ver1(plugins_json);
        }
    }
}

void Project::serializePlugins_ver1()
{
    nlohmann::json j;

    auto loaded_plugins = PluginManager::getLoadedPlugins();

    for(const auto& [path, handle]: loaded_plugins)
    {
        std::string filename = path.filename().stem().string();
        j["Plugins"].push_back(filename);
    }

    std::ofstream o(_project_path / "Plugins.json");
    o << std::setw(4) << j << std::endl;
}

void Project::deserializePlugins_ver1(const nlohmann::json& j)
{
    if(j.contains("Plugins"))
    {
        for(const auto& plugin_name: j["Plugins"])
        {
            PluginManager::loadPluginFromName(plugin_name);
        }
    }
}

}    // namespace atcg