#include <Medium/Medium.h>

#include <Scene/ComponentRegistry.h>

namespace atcg
{
namespace GUI
{

void ComponentGUIRenderer<MediumComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                           Entity entity,
                                                           MediumComponent& component) const
{
    component.medium->onImGuiRender();
}

}    // namespace GUI

void MediumComponent::registerComponent(ComponentRegistry::Registry* registry)
{
    ATCG_REGISTER_COMPONENT(registry, "Medium", MediumComponent);
}

}    // namespace atcg