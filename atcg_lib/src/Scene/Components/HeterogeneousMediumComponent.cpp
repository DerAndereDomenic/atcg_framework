#include <Scene/Components/HeterogeneousMediumComponent.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{

namespace GUI
{
void ComponentGUIRenderer<HeterogeneousMediumComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                                        Entity entity,
                                                                        HeterogeneousMediumComponent& component) const
{
#ifndef ATCG_HEADLESS
    HeterogeneousMediumComponent _component = component;

    bool updated = false;

    ImGui::Text("Density");
    auto new_handle                = displayTexture3DSelection("densitytexture3d", _component.density_grid.handle);
    updated                        = updated || (new_handle != _component.density_grid.handle);
    _component.density_grid.handle = new_handle;
    updated =
        ImGui::DragFloat("Desity Scale##texture3d", &_component.density_grid.scale, 0.01f, 0.0f, 10.0f) || updated;
    ImGui::Text("Bounding Box");
    updated = ImGui::DragFloat3("Min##density", glm::value_ptr(_component.density_grid.bbox.min), 0.05f) || updated;
    updated = ImGui::DragFloat3("Max##density", glm::value_ptr(_component.density_grid.bbox.max), 0.05f) || updated;

    ImGui::Separator();
    ImGui::Text("Albedo");
    new_handle                    = displayTexture3DSelection("albedotexture3d", _component.albedo_grid.handle);
    updated                       = updated || (new_handle != _component.albedo_grid.handle);
    _component.albedo_grid.handle = new_handle;
    updated = ImGui::DragFloat("Albedo Scale##texture3d", &_component.albedo_grid.scale, 0.01f, 0.0f, 1.0f) || updated;
    ImGui::Text("Bounding Box");
    updated = ImGui::DragFloat3("Min##albedo", glm::value_ptr(_component.albedo_grid.bbox.min), 0.05f) || updated;
    updated = ImGui::DragFloat3("Max##albedo", glm::value_ptr(_component.albedo_grid.bbox.max), 0.05f) || updated;

    ImGui::Separator();
    ImGui::Text("Emission");
    new_handle                      = displayTexture3DSelection("emissiontexture3d", _component.emission_grid.handle);
    updated                         = updated || (new_handle != _component.emission_grid.handle);
    _component.emission_grid.handle = new_handle;
    updated =
        ImGui::DragFloat("Emission Scale##texture3d", &_component.emission_grid.scale, 0.01f, 0.0f, 10.0f) || updated;
    ImGui::Text("Bounding Box");
    updated = ImGui::DragFloat3("Min##emission", glm::value_ptr(_component.emission_grid.bbox.min), 0.05f) || updated;
    updated = ImGui::DragFloat3("Max##emission", glm::value_ptr(_component.emission_grid.bbox.max), 0.05f) || updated;
    updated = ImGui::DragFloat("g##het", &_component.g, 0.01f, -1.0f, 1.0f) || updated;

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<HeterogeneousMediumComponent>>(scene, entity);
        component = _component;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT_DRAW(HeterogeneousMediumComponent);
ATCG_REGISTER_COMPONENT_STORE(HeterogeneousMediumComponent);
ATCG_REGISTER_COMPONENT_SERIALIZATION(HeterogeneousMediumComponent);
}    // namespace atcg