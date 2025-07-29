#include <Scene/ComponentGUIHandler.h>

namespace atcg
{
namespace GUI
{


void ComponentGUIRenderer<TransformComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                              Entity entity,
                                                              TransformComponent& transform) const
{
}


void ComponentGUIRenderer<CameraComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                           Entity entity,
                                                           CameraComponent& camera_component) const
{
}


void ComponentGUIRenderer<GeometryComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                             Entity entity,
                                                             GeometryComponent& component) const
{
}


void ComponentGUIRenderer<MeshRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               MeshRenderComponent& component) const
{
}


void ComponentGUIRenderer<PointRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                                Entity entity,
                                                                PointRenderComponent& component) const
{
}


void ComponentGUIRenderer<PointSphereRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                                      Entity entity,
                                                                      PointSphereRenderComponent& component) const
{
}


void ComponentGUIRenderer<EdgeRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               EdgeRenderComponent& component) const
{
}


void ComponentGUIRenderer<EdgeCylinderRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                                       Entity entity,
                                                                       EdgeCylinderRenderComponent& component) const
{
}


void ComponentGUIRenderer<InstanceRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                                   Entity entity,
                                                                   InstanceRenderComponent& component) const
{
}


void ComponentGUIRenderer<PointLightComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               PointLightComponent& component) const
{
}


void ComponentGUIRenderer<ScriptComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                           Entity entity,
                                                           ScriptComponent& _component) const
{
}

AssetHandle displayMaterialSelection(const std::string& key, AssetHandle handle)
{
    return 0;
}

AssetHandle displayGraphSelection(const std::string& key, AssetHandle handle)
{
    return 0;
}

AssetHandle displayScriptSelection(const std::string& key, AssetHandle handle)
{
    return 0;
}

AssetHandle displayShaderSelection(const std::string& key, AssetHandle handle)
{
    return 0;
}

AssetHandle displayTexture2DSelection(const std::string& key, AssetHandle handle)
{
    return 0;
}

}    // namespace GUI
}    // namespace atcg