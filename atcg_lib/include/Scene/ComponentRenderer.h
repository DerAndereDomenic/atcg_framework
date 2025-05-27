#pragma once

#include <Core/SystemRegistry.h>
#include <Scene/Scene.h>
#include <Scene/Components.h>
#include <DataStructure/Dictionary.h>
#include <DataStructure/Skybox.h>

namespace atcg
{
/**
 * @brief A class that manages how different components are rendered
 */
class ComponentRenderer
{
public:
    /**
     * @brief Default constructor
     */
    ComponentRenderer()
        : _renderer(SystemRegistry::instance()->getSystem<RendererSystem>()),
          _dummy_skybox(atcg::make_ref<Skybox>())
    {
    }

    /**
     * @brief Constructor
     *
     * @param renderer A pointer to the renderer that should be used internally. This does not take ownership of the
     * renderer and expects that the pointer stays valid while this class is in use.
     */
    ComponentRenderer(atcg::RendererSystem* renderer) : _renderer(renderer), _dummy_skybox(atcg::make_ref<Skybox>()) {}

    /**
     * @brief Render a component.
     * The auxiliary dictionary can be used to add additional information to the render process. Currently the following
     * keys are used:
     * * point_light_depth_maps: atcg::ref_ptr<atcg::TextureCubeArray> - Depth maps of the point light sources.
     * Shadowmapping is disabled if this is missing or nullptr.
     * * skybox: atcg::ref_ptr<Skybox> - A skybox used for ibl. If this is missing or nullptr, it will be replaced by a
     * dummy skybox that is completely black.
     * * has_skybox: bool - If this is true, the skybox is used for ibl, if not it will be ignored. false per default
     * * override_shader: atcg::ref_ptr<Shader> - Can be used to override the default shader of a component, for example
     * for CylinderEdgeRendererComponent. Per default nullptr.
     *
     * @tparam T The component type
     * @param entity The entity that owns the component
     * @param camera The camera
     * @param auxiliary The auxiliary data that may be used for the rendering process
     */
    template<typename T>
    void renderComponent(Entity entity, const atcg::ref_ptr<Camera>& camera, atcg::Dictionary& auxiliary);

protected:
    uint32_t _setLights(Scene* scene,
                        const atcg::ref_ptr<atcg::TextureCubeArray>& point_light_depth_maps,
                        const atcg::ref_ptr<Shader>& shader);

    std::pair<uint32_t, uint32_t> _setSkyLight(const atcg::ref_ptr<Shader>& shader,
                                               const atcg::ref_ptr<Skybox>& skybox);

private:
    RendererSystem* _renderer;
    atcg::ref_ptr<Skybox> _dummy_skybox;
};
}    // namespace atcg