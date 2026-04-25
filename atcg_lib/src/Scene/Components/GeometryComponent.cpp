#include <Scene/Components/GeometryComponent.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{

namespace GUI
{
void ComponentGUIRenderer<GeometryComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                             Entity entity,
                                                             GeometryComponent& component) const
{
#ifndef ATCG_HEADLESS
    auto new_handle = displayGraphSelection("geometry", component.graph_handle);
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