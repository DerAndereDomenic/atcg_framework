#pragma once

#include <Core/API.h>
#include <Core/Memory.h>
#include <Core/RaytracingContext.h>
#include <Integrator/Integrator.h>
#include <Integrator/IntegratorSettings.h>
#include <Renderer/Framebuffer.h>
#include <Renderer/Texture.h>
#include <Scene/Scene.h>

namespace atcg
{
/**
 * @brief A class that manages the lifecycle of an integrator for an application.
 *
 * The controller creates the integrator from the integrator settings of the active project, renders the scene with it
 * and keeps it up to date. It rebuilds the integrator if the scene, the settings, the target size or the scene itself
 * (detected through the revision stack) changed, and resets the accumulation if the camera moved. GUI panels (see
 * GUI::IntegratorPanel) only send commands to the controller, so the same controller can be used with or without a GUI.
 */
class ATCG_API IntegratorController
{
public:
    /**
     * @brief Constructor
     *
     * @param context The raytracing context. If it is nullptr, a new context is created.
     */
    IntegratorController(const atcg::ref_ptr<RaytracingContext>& context = nullptr);

    /**
     * @brief Destructor
     */
    ~IntegratorController();

    /**
     * @brief Start path tracing. The integrator is (re)built on the next call to render.
     * Whenever the active project has no integrator settings (on construction, on start and on render), settings for
     * the first registered integrator are created.
     */
    void start();

    /**
     * @brief Stop path tracing. The integrator is kept for a quick restart.
     */
    void stop();

    /**
     * @brief Check if path tracing is running
     *
     * @return True if running
     */
    bool isRunning() const;

    /**
     * @brief Change the integrator type.
     * New default settings for this type are created and stored in the active project.
     *
     * @param type The name of the integrator in the IntegratorRegistry
     */
    void setIntegratorType(const std::string& type);

    /**
     * @brief Get the type of the current integrator
     *
     * @return The integrator type or an empty string if there are no integrator settings
     */
    std::string getIntegratorType() const;

    /**
     * @brief Get the integrator settings of the active project
     *
     * @return The settings or nullptr if there is no active project or no settings
     */
    atcg::ref_ptr<IntegratorSettingsBase> getSettings() const;

    /**
     * @brief Rebuild the integrator on the next call to render, for example after the settings were edited
     */
    void requestRebuild();

    /**
     * @brief Reset the accumulation of the integrator but keep the integrator
     */
    void reset();

    /**
     * @brief Release the integrator. This should be called before unloading a plugin that defines the integrator.
     * The integrator is rebuilt on the next call to render if path tracing is running.
     */
    void release();

    /**
     * @brief Get the current integrator
     *
     * @return The integrator or nullptr if it was not built yet
     */
    atcg::ref_ptr<Integrator> getIntegrator() const;

    /**
     * @brief Path trace the scene and draw the result (including the entity ids) into the target framebuffer.
     * Does nothing if path tracing is not running.
     *
     * @param scene The scene to render. Its main camera (Scene::getCamera) is used.
     * @param target The target framebuffer
     *
     * @return True if the scene was path traced, false if path tracing is not running or failed
     */
    bool render(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Framebuffer>& target);

    /**
     * @brief Get the texture holding the last path traced image
     *
     * @return The output texture
     */
    atcg::ref_ptr<Texture2D> getOutputTexture() const;

    /**
     * @brief Get the texture holding the entity ids of the last path traced image
     *
     * @return The entity id texture
     */
    atcg::ref_ptr<Texture2D> getOutputEntityTexture() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg
