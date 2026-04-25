#include <Scene/Components/ScriptComponent.h>
#include <Scene/ComponentRegistry.h>

#define SCRIPT_KEY "Script"

namespace atcg
{

namespace Serialization
{
void ComponentSerializer<ScriptComponent>::serialize_component(const std::string& file_path,
                                                               const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               ScriptComponent& component,
                                                               nlohmann::json& j) const
{
    j[SCRIPT_KEY] = (uint64_t)component.script_handle;
}

void ComponentSerializer<ScriptComponent>::deserialize_component(const std::string& file_path,
                                                                 const atcg::ref_ptr<Scene>& scene,
                                                                 Entity entity,
                                                                 nlohmann::json& j) const
{
    if(!j.contains(SCRIPT_KEY))
    {
        return;
    }

    auto& script = entity.addComponent<ScriptComponent>();

    script.script_handle = (AssetHandle)j[SCRIPT_KEY];

    if(!script.script())
    {
        return;
    }

    auto behavior = script.behavior(scene, entity);

    if(behavior) behavior->onAttach();
}
}    // namespace Serialization

namespace GUI
{
void ComponentGUIRenderer<ScriptComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                           Entity entity,
                                                           ScriptComponent& _component) const
{
#ifndef ATCG_HEADLESS
    auto new_handle = displayScriptSelection("script", _component.script_handle);
    bool updated    = (new_handle != _component.script_handle);

    if(updated)
    {
        RevisionStack::startRecording<ComponentEditedRevision<ScriptComponent>>(scene, entity);
        if(_component.script())
        {
            auto behavior = _component.behavior(scene, entity);
            if(behavior)
            {
                behavior->onDetach();
            }
        }
        _component.script_handle = new_handle;
        if(_component.script())
        {
            auto behavior = _component.behavior(scene, entity, true);
            if(behavior)
            {
                behavior->onAttach();
            }
        }
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT(ScriptComponent);
}    // namespace atcg