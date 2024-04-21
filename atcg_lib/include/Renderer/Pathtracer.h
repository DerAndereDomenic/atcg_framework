#pragma once

#include <Core/Memory.h>
#include <Scene/Scene.h>
#include <Renderer/PerspectiveCamera.h>
#include <Renderer/Texture.h>
#include <memory>

namespace atcg
{
class Pathtracer
{
public:
    static void init();

    static void destroy()
    {
        stop();
        delete s_pathtracer;
    }

    static void bakeScene(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<PerspectiveCamera>& camera);

    static void start();

    static void stop();

    static void reset(const atcg::ref_ptr<PerspectiveCamera>& camera);

    static atcg::ref_ptr<Texture2D> getOutputTexture();

    static void resize(const uint32_t width, const uint32_t height);

    ~Pathtracer();

private:
    Pathtracer();

    static Pathtracer* s_pathtracer;

    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg