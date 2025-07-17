#pragma once

#include <Scene/Scene.h>
#include <Scene/Components.h>

#include <Renderer/Framebuffer.h>

namespace atcg
{
namespace GUI
{

/**
 * @brief A class that handles the rendering of gui components
 *
 * To add custom rendering code, create a class that specializes this class and add the rendering code for the custom
 * component.
 *
 * @code{.cpp}
 * template<>
 * struct atcg::GUI::ComponentGUIRenderer<CustomComponent>
 * {
 *     void draw_component(const atcg::ref_ptr<Scene>& scene, Entity entity, CustomComponent& component) const
 *     {
 *         // Render Code
 *     }
 * };
 * @endcode
 */
template<typename T>
struct ComponentGUIRenderer
{
    /**
     * @brief Draw the component
     *
     * @param scene The scene the entity belongs to
     * @param entity The entity that holds the component
     * @param component The component to render
     */
    void draw_component(const atcg::ref_ptr<Scene>& scene, Entity entity, T& component) const
    {
        throw std::logic_error("No ComponentGUIRenderer specialization available for this component type");
    }
};

#define ATCG_DECLARE_COMPONENT_GUI_RENDERER(ComponentType)                                                             \
    template<>                                                                                                         \
    struct ComponentGUIRenderer<ComponentType>                                                                         \
    {                                                                                                                  \
        void draw_component(const atcg::ref_ptr<Scene>& scene, Entity entity, ComponentType& component) const;         \
    }

ATCG_DECLARE_COMPONENT_GUI_RENDERER(TransformComponent);
ATCG_DECLARE_COMPONENT_GUI_RENDERER(CameraComponent);
ATCG_DECLARE_COMPONENT_GUI_RENDERER(GeometryComponent);
ATCG_DECLARE_COMPONENT_GUI_RENDERER(MeshRenderComponent);
ATCG_DECLARE_COMPONENT_GUI_RENDERER(PointRenderComponent);
ATCG_DECLARE_COMPONENT_GUI_RENDERER(PointSphereRenderComponent);
ATCG_DECLARE_COMPONENT_GUI_RENDERER(EdgeRenderComponent);
ATCG_DECLARE_COMPONENT_GUI_RENDERER(EdgeCylinderRenderComponent);
ATCG_DECLARE_COMPONENT_GUI_RENDERER(InstanceRenderComponent);
ATCG_DECLARE_COMPONENT_GUI_RENDERER(PointLightComponent);
ATCG_DECLARE_COMPONENT_GUI_RENDERER(ScriptComponent);

AssetHandle displayMaterialSelection(const std::string& key, AssetHandle handle);

AssetHandle displayGraphSelection(const std::string& key, AssetHandle handle);

AssetHandle displayScriptSelection(const std::string& key, AssetHandle handle);

AssetHandle displayShaderSelection(const std::string& key, AssetHandle handle);

AssetHandle displayTexture2DSelection(const std::string& key, AssetHandle handle);
}    // namespace GUI
}    // namespace atcg