#include <Emitter/Emitter.h>

#include <Scene/ComponentRegistry.h>

namespace atcg
{
namespace GUI
{

void ComponentGUIRenderer<EmitterComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                            Entity entity,
                                                            EmitterComponent& component) const
{
    component.emitter->onImGuiRender();
}
ATCG_REGISTER_COMPONENT_DRAW(EmitterComponent);
}    // namespace GUI
}    // namespace atcg