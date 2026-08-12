#include <BSDF/BSDF.h>

#include <Scene/ComponentRegistry.h>

namespace atcg
{
namespace GUI
{

void ComponentGUIRenderer<BSDFComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                         Entity entity,
                                                         BSDFComponent& component) const
{
    component.bsdf->onImGuiRender();
}
ATCG_REGISTER_COMPONENT_DRAW(BSDFComponent);
}    // namespace GUI
}    // namespace atcg