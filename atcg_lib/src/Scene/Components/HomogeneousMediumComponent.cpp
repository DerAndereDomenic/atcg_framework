#include <Scene/Components/HomogeneousMediumComponent.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{

namespace GUI
{
void ComponentGUIRenderer<HomogeneousMediumComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                                      Entity entity,
                                                                      HomogeneousMediumComponent& component) const
{
#ifndef ATCG_HEADLESS
    HomogeneousMediumComponent _component = component;

    bool updated = false;
    updated      = ImGui::DragFloat("Density##homogen", &_component.density, 0.05f, 0.0f, 50.0f) || updated;
    updated      = ImGui::DragFloat("g##homogen", &_component.g, 0.01f, -1.0f, 1.0f) || updated;
    updated      = ImGui::ColorEdit3("albedo##homogen", glm::value_ptr(_component.albedo)) || updated;
    updated      = ImGui::DragFloat("Le##homogen", &_component.Le, 0.01f, 0.0f, 100.0f) || updated;
    updated      = ImGui::ColorEdit3("LeColor##homogen", glm::value_ptr(_component.Le_color)) || updated;

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<HomogeneousMediumComponent>>(scene, entity);
        component = _component;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT_DRAW(HomogeneousMediumComponent);
ATCG_REGISTER_COMPONENT_STORE(HomogeneousMediumComponent);
ATCG_REGISTER_COMPONENT_SERIALIZATION(HomogeneousMediumComponent);
}    // namespace atcg