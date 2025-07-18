#pragma once

#include <Core/Memory.h>
#include <Scene/Scene.h>

#include <json.hpp>

#include <filesystem>

namespace atcg
{

class Project
{
public:
    static atcg::ref_ptr<Project> create(const std::filesystem::path& path);

    static atcg::ref_ptr<Project> load(const std::filesystem::path& path);

    void save();

    void setActiveScene(const atcg::ref_ptr<Scene>& scene);

    void setActiveScene(AssetHandle handle);

    atcg::ref_ptr<Scene> getActiveScene() const;

    static const atcg::ref_ptr<Project>& getActive();

    static void saveActive();

    ATCG_INLINE const std::filesystem::path& getFilePath() const { return _project_path; }

private:
    void serializeProjectInformation();
    void serializeProjectInformation_ver1();

    void deserializeProjectInformation();
    void deserializeProjectInformation_ver1(const nlohmann::json& j);

private:
    std::filesystem::path _project_path;
    std::filesystem::path _asset_pack_directory;
    AssetHandle _active_scene;

    inline static atcg::ref_ptr<Project> s_active_project = nullptr;
};
}    // namespace atcg