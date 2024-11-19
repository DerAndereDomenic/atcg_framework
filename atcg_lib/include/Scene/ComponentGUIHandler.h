#pragma once

#include <Scene/Scene.h>
#include <Scene/Components.h>

#include <Renderer/Framebuffer.h>

namespace atcg
{
/**
 * @brief A class that handles the rendering of gui components
 *
 * To add custom rendering code, create a class that inherits from this class and add the rendering code for the custom
 * component. To get the default behavior, override the draw_component function that calls the super class. Add this
 * class as template argument to the atcg::SceneHierarchyPanel.
 *
 * @note The code below is an example for MSVC. Under different compilers, one might have to restructure the template
 * specialization.
 *
 * @code{.cpp}
 * class MyComponentGUIHandler : public ComponentGUIHandler
 * {
 * public:
 *      MyComponentGUIHandler(const atcg::ref_ptr<Scene>& scene) :ComponentGUIHandler(scene) {}
 *
 *      template<typename T>
 *      void draw_component(Entity entity, T& component)
 *      {
 *          atcg::ComponentGUIHandler::draw_component<T>(entity, component);
 *      }
 *
 *      template<>
 *      void draw_component<MyCustomComponent>(Entity entity, MyCustomComponent& component)
 *      {
 *          // Draw custom component
 *      }
 * };
 * @endcode
 */
class ComponentGUIHandler
{
public:
    /**
     * @brief Constructor
     *
     * @param scene The scene
     */
    ComponentGUIHandler(const atcg::ref_ptr<Scene>& scene)
        : _scene(scene),
          _camera_preview(atcg::make_ref<atcg::Framebuffer>(128, 128))
    {
        _camera_preview->attachColor();
        _camera_preview->attachDepth();
        _camera_preview->complete();
    }

    /**
     * @brief Draw a component
     *
     * @tparam T The component type
     *
     * @param entity The entity that owns the component
     * @param component The component
     */
    template<typename T>
    void draw_component(Entity entity, T& component);

protected:
    void displayMaterial(const std::string& key, Material& material);

    atcg::ref_ptr<Scene> _scene;
    atcg::ref_ptr<Framebuffer> _camera_preview;
};
}    // namespace atcg