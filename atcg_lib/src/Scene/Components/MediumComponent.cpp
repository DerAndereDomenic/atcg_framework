#include <Scene/Components/MediumComponent.h>
#include <Scene/ComponentRegistry.h>
#include <Utils/Utils.h>

#define MEDIUM_KEY "Medium"

namespace atcg
{

void ComponentRenderer<MediumComponent>::renderComponent(atcg::RendererSystem* _renderer,
                                                         Entity entity,
                                                         const atcg::ref_ptr<Camera>& camera,
                                                         atcg::Dictionary& auxiliary) const
{
    // TODO
}

namespace Serialization
{
void ComponentSerializer<MediumComponent>::serialize_component(const std::string& file_path,
                                                               const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               MediumComponent& component,
                                                               nlohmann::json& j) const
{
    j[MEDIUM_KEY] = (uint64_t)component.medium_handle;
}

void ComponentSerializer<MediumComponent>::deserialize_component(const std::string& file_path,
                                                                 const atcg::ref_ptr<Scene>& scene,
                                                                 Entity entity,
                                                                 nlohmann::json& j) const
{
    if(!j.contains(MEDIUM_KEY))
    {
        return;
    }

    auto& medium         = entity.addComponent<MediumComponent>();
    medium.medium_handle = (AssetHandle)j[MEDIUM_KEY];
}


}    // namespace Serialization

namespace GUI
{
void ComponentGUIRenderer<MediumComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                           Entity entity,
                                                           MediumComponent& component) const
{
#ifndef ATCG_HEADLESS
    MediumComponent copy = component;
    bool deactivated     = false;
    auto new_handle      = Utils::displayMediumSelection("medium", copy.medium_handle, deactivated);
    bool updated         = (new_handle != copy.medium_handle);
    copy.medium_handle   = new_handle;

    if(updated)
    {
        RevisionStack::startRecording<ComponentEditedRevision<MediumComponent>>(scene, entity);
        component = copy;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT(MediumComponent);
}    // namespace atcg