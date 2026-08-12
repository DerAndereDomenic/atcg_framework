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

ATCG_REGISTER_COMPONENT_DRAW(PhaseFunctionComponent);
}    // namespace GUI
}    // namespace atcg