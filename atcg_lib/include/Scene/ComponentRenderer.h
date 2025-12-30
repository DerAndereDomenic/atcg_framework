#pragma once

#include <Core/SystemRegistry.h>
#include <Scene/Scene.h>
#include <Scene/Components.h>
#include <DataStructure/Dictionary.h>
#include <DataStructure/Skybox.h>

namespace atcg
{
/**
 * @brief A Template struct to manage rendering of different components.
 * @tparam T The component type
 *
 * To add custom rendering code, create a class that specializes this class and add the rendering code for the custom
 * component.
 *
 * @code{.cpp}
 * template<>
 * struct atcg::ComponentRenderer<CustomComponent>
 * {
 *     void renderComponent(atcg::RendererSystem* renderer,
 *                          Entity entity,
 *                          const atcg::ref_ptr<Camera>& camera,
 *                          atcg::Dictionary& auxiliary) const
 *     {
 *         // Render Code
 *     }
 * };
 * @endcode
 */
template<typename T>
struct ComponentRenderer
{
    /**
     * @brief Render a component.
     * The auxiliary dictionary can be used to add additional information to the render process. Currently the following
     * keys are used:
     * * point_light_depth_maps: atcg::ref_ptr<atcg::TextureCubeArray> - Depth maps of the point light sources.
     * Shadowmapping is disabled if this is missing or nullptr.
     * * skybox: atcg::ref_ptr<Skybox> - A skybox used for ibl. If this is missing or nullptr, it will be replaced by a
     * dummy skybox that is completely black.
     * * has_skybox: bool - If this is true, the skybox is used for ibl, if not it will be ignored. false per default
     * * override_shader: atcg::ref_ptr<Shader> - Can be used to override the default shader of a component, for
     example
     * for CylinderEdgeRendererComponent. Per default nullptr.
     *
     * @tparam T The component type
     * @param renderer The renderer
     * @param entity The entity that owns the component
     * @param camera The camera
     * @param auxiliary The auxiliary data that may be used for the rendering process
     */
    void renderComponent(atcg::RendererSystem* renderer,
                         Entity entity,
                         const atcg::ref_ptr<Camera>& camera,
                         atcg::Dictionary& auxiliary) const
    {
    }
};

#define ATCG_DECLARE_COMPONENT_RENDERER(ComponentType)                                                                 \
    template<>                                                                                                         \
    struct ComponentRenderer<ComponentType>                                                                            \
    {                                                                                                                  \
        void renderComponent(atcg::RendererSystem* renderer,                                                           \
                             Entity entity,                                                                            \
                             const atcg::ref_ptr<Camera>& camera,                                                      \
                             atcg::Dictionary& auxiliary) const;                                                       \
    };

ATCG_DECLARE_COMPONENT_RENDERER(MeshRenderComponent);
ATCG_DECLARE_COMPONENT_RENDERER(PointRenderComponent);
ATCG_DECLARE_COMPONENT_RENDERER(PointSphereRenderComponent);
ATCG_DECLARE_COMPONENT_RENDERER(EdgeRenderComponent);
ATCG_DECLARE_COMPONENT_RENDERER(EdgeCylinderRenderComponent);
ATCG_DECLARE_COMPONENT_RENDERER(InstanceRenderComponent);
ATCG_DECLARE_COMPONENT_RENDERER(MeshLightComponent);
ATCG_DECLARE_COMPONENT_RENDERER(CameraComponent);

template<typename T>
void renderComponent(atcg::RendererSystem* renderer,
                     Entity entity,
                     const atcg::ref_ptr<Camera>& camera,
                     atcg::Dictionary& auxiliary)
{
    if(!entity.hasComponent<T>()) return;

    ComponentRenderer<T>().renderComponent(renderer, entity, camera, auxiliary);
}
}    // namespace atcg