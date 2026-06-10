#pragma once

#include <Asset/AssetManagerSystem.h>
#include <Core/glm.h>
#include <Renderer/Camera.h>
#include <Renderer/Framebuffer.h>
#include <Renderer/PerspectiveCamera.h>
#include <Scene/Components/TransformComponent.h>
#include <Scene/ComponentGUIHandler.h>
#include <Scene/ComponentRenderer.h>
#include <Scene/ComponentSerializer.h>

namespace atcg
{
struct ATCG_API CameraComponent
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

ATCG_DECLARE_COMPONENT_RENDERER(CameraComponent);

namespace Serialization
{
ATCG_DECLARE_COMPONENT_SERIALIZER(CameraComponent);
}

namespace GUI
{

template<>
ATCG_INLINE void displayAddComponentEntry<CameraComponent>(const atcg::ref_ptr<atcg::Scene>& scene, Entity entity)
{
#ifndef ATCG_HEADLESS
    if(!entity.hasComponent<CameraComponent>())
    {
        if(ImGui::MenuItem(CameraComponent::toString()))
        {
            atcg::RevisionStack::startRecording<ComponentAddedRevision<CameraComponent>>(scene, entity);
            auto& camera_component = entity.addComponent<CameraComponent>(atcg::make_ref<PerspectiveCamera>());
            if(entity.hasComponent<TransformComponent>())
            {
                atcg::ref_ptr<PerspectiveCamera> cam =
                    std::dynamic_pointer_cast<PerspectiveCamera>(camera_component.camera);
                cam->setView(glm::inverse(entity.getComponent<TransformComponent>().getModel()));
            }
            ImGui::CloseCurrentPopup();
            atcg::RevisionStack::endRecording();
        }
    }
#endif
}

ATCG_DECLARE_COMPONENT_GUI_RENDERER(CameraComponent);
}    // namespace GUI

}    // namespace atcg