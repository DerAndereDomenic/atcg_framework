#pragma once

#include <Scene/Scene.h>
#include <Scene/Components.h>
#include <DataStructure/Dictionary.h>

namespace atcg
{
class ComponentRenderer
{
public:
    ComponentRenderer() = default;

    template<typename T>
    void renderComponent(Entity entity, const atcg::ref_ptr<Camera>& camera, atcg::Dictionary& auxiliary);

protected:
    uint32_t _setLights(Scene* scene,
                        const atcg::ref_ptr<atcg::TextureCubeArray>& point_light_depth_maps,
                        const atcg::ref_ptr<Shader>& shader);
};
}    // namespace atcg