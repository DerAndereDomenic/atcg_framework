#include <Scene/Components/GeometryComponent.h>
#include <Scene/ComponentRegistry.h>
#include <Utils/Utils.h>

#define GEOMETRY_KEY "Geometry"

namespace atcg
{

namespace Serialization
{
void ComponentSerializer<GeometryComponent>::serialize_component(const std::string& file_path,
                                                                 const atcg::ref_ptr<Scene>& scene,
                                                                 Entity entity,
                                                                 GeometryComponent& component,
                                                                 nlohmann::json& j) const
{
    j[GEOMETRY_KEY] = (uint64_t)component.graph_handle;
}

void ComponentSerializer<GeometryComponent>::deserialize_component(const std::string& file_path,
                                                                   const atcg::ref_ptr<Scene>& scene,
                                                                   Entity entity,
                                                                   nlohmann::json& j) const
{
    if(!j.contains(GEOMETRY_KEY))
    {
        return;
    }

    auto& geometry        = entity.addComponent<GeometryComponent>();
    geometry.graph_handle = (AssetHandle)j[GEOMETRY_KEY];
}


}    // namespace Serialization

namespace GUI
{
void ComponentGUIRenderer<GeometryComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                             Entity entity,
                                                             GeometryComponent& component) const
{
#ifndef ATCG_HEADLESS
    auto new_handle = Utils::displayGraphSelection("geometry", component.graph_handle);
    bool updated    = (new_handle != component.graph_handle);

    if(updated)
    {
        RevisionStack::startRecording<ComponentEditedRevision<GeometryComponent>>(scene, entity);
        component.graph_handle = new_handle;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT(GeometryComponent);
}    // namespace atcg