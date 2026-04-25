#include <Scene/Components/EdgeCylinderRenderComponent.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{
namespace GUI
{
void ComponentGUIRenderer<EdgeCylinderRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                                       Entity entity,
                                                                       EdgeCylinderRenderComponent& _component) const
{
#ifndef ATCG_HEADLESS
    EdgeCylinderRenderComponent component = _component;
    std::string id                        = std::to_string(entity.getComponent<IDComponent>().ID());

    bool updated = ImGui::Checkbox("Visible##visibleedgecylinder", &component.visible);
    std::stringstream label;
    label << "Radius##edgecylinder" << id;
    float radius = component.radius;
    if(ImGui::DragFloat(label.str().c_str(), &radius, 0.001f, 0.001f, FLT_MAX / INT_MAX))
    {
        component.radius = radius;
        updated          = true;
    }

    // Material
    auto material_handle = component.material_handle;

    auto new_handle           = displayMaterialSelection("edgecylinder", material_handle);
    updated                   = (new_handle != material_handle) || updated;
    component.material_handle = new_handle;

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<EdgeCylinderRenderComponent>>(scene, entity);
        _component = component;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT(EdgeCylinderRenderComponent);
}    // namespace atcg