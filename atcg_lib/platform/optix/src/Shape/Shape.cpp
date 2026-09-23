#include <Shape/Shape.h>

#include <Scene/ComponentRegistry.h>

namespace atcg
{
namespace GUI
{
void ComponentGUIRenderer<ShapeComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                          Entity entity,
                                                          ShapeComponent& component) const
{
    component.shape->onImGuiRender();
}
}    // namespace GUI

void ShapeComponent::registerComponent(ComponentRegistry::Registry* registry)
{
    ATCG_REGISTER_COMPONENT(registry, "Shape", ShapeComponent);
}

}    // namespace atcg