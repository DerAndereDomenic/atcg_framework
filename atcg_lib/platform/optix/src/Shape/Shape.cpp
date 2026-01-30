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
ATCG_REGISTER_COMPONENT_DRAW(ShapeComponent);
}    // namespace GUI
}    // namespace atcg