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

}    // namespace GUI

void BSDFComponent::registerComponent(ComponentRegistry::Registry* registry)
{
    ATCG_REGISTER_COMPONENT(registry, "BSDF", BSDFComponent);
}
}    // namespace atcg