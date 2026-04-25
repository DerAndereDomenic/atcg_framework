#include <Scene/Components/EdgeRenderComponent.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{

namespace GUI
{
void ComponentGUIRenderer<EdgeRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               EdgeRenderComponent& _component) const
{
#ifndef ATCG_HEADLESS
    EdgeRenderComponent component = _component;

    std::string id = std::to_string(entity.getComponent<IDComponent>().ID());

    bool updated    = ImGui::Checkbox("Visible##visibleedge", &component.visible);
    glm::vec3 color = component.color;
    std::stringstream label;
    label << "Base Color##edge" << id;
    if(ImGui::ColorEdit3(label.str().c_str(), glm::value_ptr(color)))
    {
        component.color = color;
        updated         = true;
    }

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<EdgeRenderComponent>>(scene, entity);
        _component = component;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT(EdgeRenderComponent);
}    // namespace atcg