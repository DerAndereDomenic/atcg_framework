#include <Scene/Components/PointRenderComponent.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{

namespace GUI
{
void ComponentGUIRenderer<PointRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                                Entity entity,
                                                                PointRenderComponent& _component) const
{
#ifndef ATCG_HEADLESS
    std::string id = std::to_string(entity.getComponent<IDComponent>().ID());

    PointRenderComponent component = _component;

    bool updated    = ImGui::Checkbox("Visible##visiblepoints", &component.visible);
    glm::vec3 color = component.color;
    std::stringstream label;
    label << "Base Color##point" << id;
    if(ImGui::ColorEdit3(label.str().c_str(), glm::value_ptr(color)))
    {
        component.color = color;
        updated         = true;
    }

    int point_size = (int)component.point_size;
    label.str(std::string());
    label << "Point Size##point" << id;
    if(ImGui::DragInt(label.str().c_str(), &point_size, 1, 1, INT_MAX))
    {
        component.point_size = (float)point_size;
        updated              = true;
    }

    auto shader_handle      = component.shader_handle;
    auto new_handle         = displayShaderSelection("point", shader_handle);
    updated                 = (new_handle != shader_handle) || updated;
    component.shader_handle = new_handle;

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<PointRenderComponent>>(scene, entity);
        _component = component;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT(PointRenderComponent);
}    // namespace atcg