#include <Scene/Components/MeshRenderComponent.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{

namespace GUI
{
void ComponentGUIRenderer<MeshRenderComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               MeshRenderComponent& component) const
{
#ifndef ATCG_HEADLESS
    MeshRenderComponent component_copy = component;
    bool updated                       = ImGui::Checkbox("Visible##visiblemesh", &component_copy.visible);

    // Material
    auto material_handle = component_copy.material_handle;
    auto shader_handle   = component_copy.shader_handle;

    auto new_handle                = displayMaterialSelection("mesh", material_handle);
    updated                        = (new_handle != material_handle) || updated;
    component_copy.material_handle = new_handle;

    new_handle                   = displayShaderSelection("mesh", shader_handle);
    updated                      = (new_handle != shader_handle) || updated;
    component_copy.shader_handle = new_handle;

    updated = ImGui::Checkbox("Receive Shadows##MeshRenderComponent", &component_copy.receive_shadow) || updated;

    if(updated)
    {
        atcg::RevisionStack::startRecording<ComponentEditedRevision<MeshRenderComponent>>(scene, entity);
        component = component_copy;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT(MeshRenderComponent);
}    // namespace atcg