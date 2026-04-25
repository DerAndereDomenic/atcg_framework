#include <Scene/Components/InstanceRenderComponent.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{
namespace GUI
{
void ComponentGUIRenderer<InstanceRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                                   Entity entity,
                                                                   InstanceRenderComponent& _component) const
{
#ifndef ATCG_HEADLESS
    InstanceRenderComponent component = _component;
    bool updated                      = ImGui::Checkbox("Visible##visibleinstance", &component.visible);

    // Material
    auto material_handle = component.material_handle;

    auto new_handle           = displayMaterialSelection("instance", material_handle);
    updated                   = (new_handle != material_handle) || updated;
    component.material_handle = new_handle;
    updated = ImGui::Checkbox("Receive Shadows##InstanceRenderComponent", &component.receive_shadow) || updated;

    auto shader_handle      = component.shader_handle;
    new_handle              = displayShaderSelection("instance", shader_handle);
    updated                 = (new_handle != shader_handle) || updated;
    component.shader_handle = new_handle;

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<InstanceRenderComponent>>(scene, entity);
        _component = component;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT(InstanceRenderComponent);
}    // namespace atcg