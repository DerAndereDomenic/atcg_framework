#include <Asset/Project.h>
#include <Asset/AssetManagerSystem.h>

#include <json.hpp>

namespace atcg
{
atcg::ref_ptr<Project> Project::create(const std::filesystem::path& path)
{
    s_active_project                = atcg::make_ref<Project>();
    s_active_project->_project_path = path;

    return s_active_project;
}

atcg::ref_ptr<Project> Project::load(const std::filesystem::path& path)
{
    // TODO: load
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
    AssetManager::serializeRegistry(_project_path / "AssetPack.json");

    // 3. Serialize Assets
    AssetManager::serializeAssets(_project_path / "assets");

    // 4. Serialize Project information
    serializeProjectInformation();
}

const atcg::ref_ptr<Project>& Project::getActive()
{
    return s_active_project;
}

void Project::saveActive()
{
    if(s_active_project) s_active_project->save();
}

void Project::serializeProjectInformation()
{
    nlohmann::json j;

    j["Version"] = "1.0";

    j["Assets"] = "AssetPack.json";

    std::ofstream o(_project_path / "Project.json");
    o << std::setw(4) << j << std::endl;
}
}    // namespace atcg