#include <Scene/Components/TransparencyComponent.h>
#include <Scene/ComponentRegistry.h>

#define TRANSPARENCY_KEY "Transparency"

namespace atcg
{
namespace Serialization
{
void ComponentSerializer<TransparencyComponent>::serialize_component(const std::string& file_path,
                                                                     const atcg::ref_ptr<Scene>& scene,
                                                                     Entity entity,
                                                                     TransparencyComponent& component,
                                                                     nlohmann::json& j) const
{
    j[TRANSPARENCY_KEY] = component.transparent;
}

void ComponentSerializer<TransparencyComponent>::deserialize_component(const std::string& file_path,
                                                                       const atcg::ref_ptr<Scene>& scene,
                                                                       Entity entity,
                                                                       nlohmann::json& j) const
{
    if(!j.contains(TRANSPARENCY_KEY))
    {
        return;
    }

    entity.addOrReplaceComponent<TransparencyComponent>(j[TRANSPARENCY_KEY]);
}

}    // namespace Serialization

namespace GUI
{
void ComponentGUIRenderer<TransparencyComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                                 Entity entity,
                                                                 TransparencyComponent& component) const
{
#ifndef ATCG_HEADLESS
    TransparencyComponent copy = component;

    bool updated = ImGui::Checkbox("Transparent", &copy.transparent);

    if(updated)
    {
        RevisionStack::startRecording<ComponentEditedRevision<TransparencyComponent>>(scene, entity);
        component = copy;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

void TransparencyComponent::registerComponent(ComponentRegistry::Registry* registry)
{
    ATCG_REGISTER_COMPONENT(registry, "Transparency", TransparencyComponent);
}

}    // namespace atcg