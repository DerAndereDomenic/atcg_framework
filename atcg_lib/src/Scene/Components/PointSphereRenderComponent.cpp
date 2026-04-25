#include <Scene/Components/PointSphereRenderComponent.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{

namespace GUI
{
void ComponentGUIRenderer<PointSphereRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                                      Entity entity,
                                                                      PointSphereRenderComponent& _component) const
{
#ifndef ATCG_HEADLESS
    PointSphereRenderComponent component = _component;
    std::string id                       = std::to_string(entity.getComponent<IDComponent>().ID());

    bool updated = ImGui::Checkbox("Visible##visiblepointsphere", &component.visible);

    float point_size = component.point_size;
    std::stringstream label;
    label << "Point Size##pointsphere" << id;
    if(ImGui::DragFloat(label.str().c_str(), &point_size, 0.001f, 0.001f, FLT_MAX / INT_MAX))
    {
        component.point_size = point_size;
        updated              = true;
    }

    // Material
    auto material_handle = component.material_handle;

    auto new_handle           = displayMaterialSelection("pointsphere", material_handle);
    updated                   = (new_handle != material_handle) || updated;
    component.material_handle = new_handle;

    auto shader_handle      = component.shader_handle;
    new_handle              = displayShaderSelection("pointsphere", shader_handle);
    updated                 = (new_handle != shader_handle) || updated;
    component.shader_handle = new_handle;

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<PointSphereRenderComponent>>(scene, entity);
        _component = component;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT(PointSphereRenderComponent);
}    // namespace atcg