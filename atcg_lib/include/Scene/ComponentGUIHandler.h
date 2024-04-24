#pragma once

#include <Scene/Scene.h>
#include <Scene/Components.h>

#include <Renderer/Framebuffer.h>

namespace atcg
{
class ComponentGUIHandler
{
public:
    ComponentGUIHandler(const atcg::ref_ptr<Scene>& scene)
        : _scene(scene),
          _camera_preview(atcg::make_ref<atcg::Framebuffer>(128, 128))
    {
        _camera_preview->attachColor();
        _camera_preview->attachDepth();
        _camera_preview->complete();
    }

    template<typename T>
    void draw_component(Entity entity, T& component);

protected:
    void displayMaterial(const std::string& key, Material& material);

    atcg::ref_ptr<Scene> _scene;
    atcg::ref_ptr<Framebuffer> _camera_preview;
};
}    // namespace atcg