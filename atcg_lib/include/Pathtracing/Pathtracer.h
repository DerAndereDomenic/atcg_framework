#pragma once

#include <Core/Memory.h>
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
class Pathtracer
{
public:
    /**
     * @brief Initializes the pathtracer.
     */
    static void init();

    /**
     * @brief Destroy the pathtracing instance.
     */
    static void destroy();

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
    static void draw(const atcg::ref_ptr<Scene>& scene,
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
    static void draw(const atcg::ref_ptr<Scene>& scene,
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
    static bool isRunning();

    /**
     * @brief Stop the current rendering job
     */
    static void stop();

    /**
     * @brief Get the output texture
     *
     * @return The output texture
     */
    static atcg::ref_ptr<Texture2D> getOutputTexture();

    /**
     * @brief Get the current frame index
     *
     * @return The frame index
     */
    static uint32_t getFrameIndex();

    /**
     * @brief Get the current output width
     *
     * @return The width
     */
    static uint32_t getWidth();

    /**
     * @brief Get the current output height
     *
     * @return The height
     */
    static uint32_t getHeight();

    /**
     * @brief Get the rendering time of the last rendering job
     *
     * @return The rendering time
     */
    static float getLastRenderingTime();

    /**
     * @brief Get the number of samples
     *
     * @return The number of samples
     */
    static uint32_t getSampleCount();

    ~Pathtracer();

private:
    Pathtracer();

    static Pathtracer* s_pathtracer;

    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg