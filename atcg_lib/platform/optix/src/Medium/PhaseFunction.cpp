#include <Medium/PhaseFunction.h>

#include <Scene/ComponentRegistry.h>

namespace atcg
{
namespace GUI
{
void ComponentGUIRenderer<PhaseFunctionComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                                  Entity entity,
                                                                  PhaseFunctionComponent& component) const
{
    component.phase_function->onImGuiRender();
}

}    // namespace GUI

void PhaseFunctionComponent::registerComponent(ComponentRegistry::Registry* registry)
{
    ATCG_REGISTER_COMPONENT(registry, "PhaseFunction", PhaseFunctionComponent);
}

}    // namespace atcg