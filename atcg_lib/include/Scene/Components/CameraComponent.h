#pragma once

#include <Asset/AssetManagerSystem.h>
#include <Core/glm.h>
#include <Renderer/Camera.h>
#include <Renderer/Framebuffer.h>
#include <Renderer/PerspectiveCamera.h>

namespace atcg
{
struct CameraComponent
{
    CameraComponent() = default;
    CameraComponent(const atcg::ref_ptr<Camera>& camera, const uint32_t width = 1024, const uint32_t height = 1024)
        : camera(camera),
          width(width),
          height(height)
    {
        if(dynamic_cast<PerspectiveCamera*>(camera.get()))
        {
            perspective = true;
        }
    }

    ATCG_INLINE atcg::ref_ptr<Texture2D> image() const
    {
        auto image = AssetManager::getAsset<Texture2D>(image_handle);
        return image;
    }

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Camera"; }

    atcg::ref_ptr<Camera> camera;
    glm::vec3 color    = glm::vec3(1);
    bool perspective   = false;
    uint32_t width     = 1024;
    uint32_t height    = 1024;
    float render_scale = 1.0f;

    atcg::ref_ptr<atcg::Framebuffer> preview;
    AssetHandle image_handle;
    bool render_preview = false;
};
}    // namespace atcg