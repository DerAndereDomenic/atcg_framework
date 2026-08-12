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
ATCG_REGISTER_COMPONENT_DRAW(MediumComponent);
}    // namespace GUI
}    // namespace atcg