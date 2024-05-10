#pragma once

#include <Core/Memory.h>
#include <Scene/Scene.h>
#include <Renderer/PerspectiveCamera.h>
#include <Renderer/Texture.h>
#include <Pathtracing/RaytracingShader.h>
#include <memory>

namespace atcg
{
class Pathtracer
{
public:
    static void init();

    static void destroy();

    static void draw(const atcg::ref_ptr<Scene>& scene,
                     const atcg::ref_ptr<PerspectiveCamera>& camera,
                     const atcg::ref_ptr<RaytracingShader>& shader,
                     uint32_t width,
                     uint32_t height,
                     const uint32_t num_samples);

    // static void draw(const atcg::ref_ptr<Scene>& scene,
    //                  const atcg::ref_ptr<PerspectiveCamera>& camera,
    //                  const atcg::ref_ptr<RaytracingShader>& shader,
    //                  uint32_t width,
    //                  uint32_t height,
    //                  const float time);

    static bool isRunning();

    static void stop();

    static atcg::ref_ptr<Texture2D> getOutputTexture();

    static uint32_t getFrameIndex();

    static uint32_t getWidth();

    static uint32_t getHeight();

    ~Pathtracer();

private:
    Pathtracer();

    static Pathtracer* s_pathtracer;

    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg