#pragma once

#include <Core/Memory.h>
#include <Scene/Scene.h>

#include <json.hpp>

#include <filesystem>

namespace atcg
{

/**
 * @brief A class to model a project.
 *
 * A project handles the high level serialization and deserialization of assets and scenes.
 */
class Project
{
public:
    /**
     * @brief Create a new project.
     * This also sets the currently active project
     *
     * @param path The root path of the project
     *
     * @return The project
     */
    static atcg::ref_ptr<Project> create(const std::filesystem::path& path);

    /**
     * @brief Load a project
     * This also sets the currently active project
     *
     * @param path The path
     *
     * @return The project
     */
    static atcg::ref_ptr<Project> load(const std::filesystem::path& path);

    /**
     * @brief Safe the project
     */
    void save();

    /**
     * @brief Save theproject at a specific path
     * This changes the file path of the project
     *
     * @param path The new path of the project
     */
    void save(const std::filesystem::path& path);

    /**
     * @brief Set the active scene.
     * If the scene is not already an registered asset, it will be registered
     *
     * @param scene The scene
     */
    void setActiveScene(const atcg::ref_ptr<Scene>& scene);

    /**
     * @brief Set the active scene.
     *
     * @param handle The handle
     */
    void setActiveScene(AssetHandle handle);

    /**
     * @brief Get the active scene
     *
     * @return The active scene
     */
    atcg::ref_ptr<Scene> getActiveScene() const;

    /**
     * @brief Get the currently active project
     *
     * @return The project
     */
    static const atcg::ref_ptr<Project>& getActive();

    /**
     * @brief Save the currently active project
     */
    static void saveActive();

    /**
     * @brief Save the currently active project at a specific path
     * This changes the file path of the project
     *
     * @param path The new path of the project
     */
    static void saveActive(const std::filesystem::path& path);

    /**
     * @brief Get the path to the project
     *
     * @return The path
     */
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