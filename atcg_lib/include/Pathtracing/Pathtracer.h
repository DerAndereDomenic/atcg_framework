#pragma once

#include <Core/Memory.h>
#include <Core/SystemRegistry.h>
#include <Scene/Scene.h>
#include <Renderer/PerspectiveCamera.h>
#include <Renderer/Texture.h>
#include <Pathtracing/RaytracingShader.h>
#include <memory>

namespace atcg
{
/**
 * @brief A module for a pathtracer.
 * This module runs on a separate thread.
 */
class PathtracingSystem
{
public:
    /**
     * @brief Initializes the pathtracer.
     */
    void init();

    /**
     * @brief Start a draw call. This runs on a separate thread. Use isRunning() to wait for the result.
     *
     * @param scene The scene
     * @param camera The camera
     * @param shader The shader
     * @param width The output width
     * @param height The output height
     * @param num_samples The number of samples
     */
    void draw(const atcg::ref_ptr<Scene>& scene,
              const atcg::ref_ptr<PerspectiveCamera>& camera,
              const atcg::ref_ptr<RaytracingShader>& shader,
              uint32_t width,
              uint32_t height,
              const uint32_t num_samples);

    /**
     * @brief Start a draw call. This runs on a separate thread. Use isRunning() to wait for the result.
     *
     * @param scene The scene
     * @param camera The camera
     * @param shader The shader
     * @param width The output width
     * @param height The output height
     * @param time The rendering time
     */
    void draw(const atcg::ref_ptr<Scene>& scene,
              const atcg::ref_ptr<PerspectiveCamera>& camera,
              const atcg::ref_ptr<RaytracingShader>& shader,
              uint32_t width,
              uint32_t height,
              const float time);

    /**
     * @brief Check if the pathtracer is running
     *
     * @return True if an image is rendered
     */
    bool isRunning() const;

    /**
     * @brief Stop the current rendering job
     */
    void stop();

    /**
     * @brief Get the output texture
     *
     * @return The output texture
     */
    atcg::ref_ptr<Texture2D> getOutputTexture();

    /**
     * @brief Get the current frame index
     *
     * @return The frame index
     */
    uint32_t getFrameIndex() const;

    /**
     * @brief Get the current output width
     *
     * @return The width
     */
    uint32_t getWidth() const;

    /**
     * @brief Get the current output height
     *
     * @return The height
     */
    uint32_t getHeight() const;

    /**
     * @brief Get the rendering time of the last rendering job
     *
     * @return The rendering time
     */
    float getLastRenderingTime() const;

    /**
     * @brief Get the number of samples
     *
     * @return The number of samples
     */
    uint32_t getSampleCount() const;

    PathtracingSystem();

    ~PathtracingSystem();

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

namespace Pathtracer
{
ATCG_INLINE void init()
{
    SystemRegistry::instance()->getSystem<PathtracingSystem>()->init();
}

ATCG_INLINE void draw(const atcg::ref_ptr<Scene>& scene,
                      const atcg::ref_ptr<PerspectiveCamera>& camera,
                      const atcg::ref_ptr<RaytracingShader>& shader,
                      uint32_t width,
                      uint32_t height,
                      const uint32_t num_samples)
{
    SystemRegistry::instance()->getSystem<PathtracingSystem>()->draw(scene, camera, shader, width, height, num_samples);
}

ATCG_INLINE void draw(const atcg::ref_ptr<Scene>& scene,
                      const atcg::ref_ptr<PerspectiveCamera>& camera,
                      const atcg::ref_ptr<RaytracingShader>& shader,
                      uint32_t width,
                      uint32_t height,
                      const float time)
{
    SystemRegistry::instance()->getSystem<PathtracingSystem>()->draw(scene, camera, shader, width, height, time);
}

ATCG_INLINE bool isRunning()
{
    return SystemRegistry::instance()->getSystem<PathtracingSystem>()->isRunning();
}

ATCG_INLINE void stop()
{
    SystemRegistry::instance()->getSystem<PathtracingSystem>()->stop();
}

ATCG_INLINE atcg::ref_ptr<Texture2D> getOutputTexture()
{
    return SystemRegistry::instance()->getSystem<PathtracingSystem>()->getOutputTexture();
}

ATCG_INLINE uint32_t getFrameIndex()
{
    return SystemRegistry::instance()->getSystem<PathtracingSystem>()->getFrameIndex();
}

ATCG_INLINE uint32_t getWidth()
{
    return SystemRegistry::instance()->getSystem<PathtracingSystem>()->getWidth();
}

ATCG_INLINE uint32_t getHeight()
{
    return SystemRegistry::instance()->getSystem<PathtracingSystem>()->getHeight();
}

ATCG_INLINE float getLastRenderingTime()
{
    return SystemRegistry::instance()->getSystem<PathtracingSystem>()->getLastRenderingTime();
}

ATCG_INLINE uint32_t getSampleCount()
{
    return SystemRegistry::instance()->getSystem<PathtracingSystem>()->getSampleCount();
}
}    // namespace Pathtracer

}    // namespace atcg