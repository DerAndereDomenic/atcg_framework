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

}    // namespace GUI

void EmitterComponent::registerComponent(ComponentRegistry::Registry* registry)
{
    ATCG_REGISTER_COMPONENT(registry, "Emitter", EmitterComponent);
}

}    // namespace atcg