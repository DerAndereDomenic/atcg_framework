#pragma once

#include <Core/Memory.h>
#include <Core/Platform.h>
#include <Events/Event.h>

#include <filesystem>

#include <Scene/Entity.h>

namespace atcg
{
/**
 * @brief The interface to model a script
 */
class Script
{
public:
    /**
     * @brief Create a new script
     *
     * @param file_path The path to the script
     */
    Script(const std::filesystem::path& file_path) : _file_path(file_path) {};

    /**
     * @brief Destructor
     */
    virtual ~Script() {};

    /**
     * @brief Initializes the script. This has to be called before any of the functions below.
     *
     * @param scene The scene in which the entity resides that is attached to this script
     * @param entity The entity that is attached to this script
     */
    virtual void init(const atcg::ref_ptr<atcg::Scene>& scene, const atcg::Entity& entity) = 0;

    /**
     * @brief The function that is called when the script gets attached
     */
    virtual void onAttach() = 0;

    /**
     * @brief The function that is called in the update loop
     *
     * @param delta_time The time between consecutive frames in secodns.
     */
    virtual void onUpdate(const float delta_time) = 0;

    /**
     * @brief The function that is called on events
     *
     * @param event The event to handle
     */
    virtual void onEvent(atcg::Event* event) = 0;

    /**
     * @brief The function that is called if the script is detached
     */
    virtual void onDetach() = 0;

    /**
     * @brief Reloads the script
     */
    virtual void reload() = 0;

    /**
     * @brief Get the file path of the underlying script
     *
     * @return The file path
     */
    ATCG_INLINE const std::filesystem::path& getFilePath() const { return _file_path; }

protected:
    std::filesystem::path _file_path;
};

/**
 * @brief A class to model a python script.
 * An instance of this function can only be used if a ScriptEngine exists and is initialized.
 */
class PythonScript : public Script
{
public:
    /**
     * @brief Create a new script
     *
     * @param file_path The path to the script
     */
    PythonScript(const std::filesystem::path& file_path);

    /**
     * @brief Destructor
     */
    virtual ~PythonScript();

    /**
     * @brief Initializes the script. This has to be called before any of the functions below.
     *
     * @param scene The scene in which the entity resides that is attached to this script
     * @param entity The entity that is attached to this script
     */
    virtual void init(const atcg::ref_ptr<atcg::Scene>& scene, const atcg::Entity& entity) override;

    /**
     * @brief The function that is called when the script gets attached
     */
    virtual void onAttach() override;

    /**
     * @brief The function that is called in the update loop
     *
     * @param delta_time The time between consecutive frames in secodns.
     */
    virtual void onUpdate(const float delta_time) override;

    /**
     * @brief The function that is called on events
     *
     * @param event The event to handle
     */
    virtual void onEvent(atcg::Event* event) override;

    /**
     * @brief The function that is called if the script is detached
     */
    virtual void onDetach() override;

    /**
     * @brief Reloads the script
     */
    virtual void reload() override;

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

namespace Scripting
{
/**
 * @brief This function goes through all scripts in the given scene and reloads them
 *
 * @param scene The scene
 */
void handleScriptReloads(const atcg::ref_ptr<atcg::Scene>& scene);

/**
 * @brief This function should be called inside the main event loop and passes it to the scripts bound to entities
 * inside the given scene.
 *
 * @param scene The scene
 * @param event The event to handle
 */
void handleScriptEvents(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Event* event);

/**
 * @brief This function should be called inside the main update loop and calls the update function to all scripts bound
 * to entities inside the given scene.
 *
 * @param scene The scene
 * @param dt The time between two frames in seconds
 */
void handleScriptUpdates(const atcg::ref_ptr<atcg::Scene>& scene, const float dt);
}    // namespace Scripting

}    // namespace atcg