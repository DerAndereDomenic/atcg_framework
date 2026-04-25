#include <Scene/Components/ScriptComponent.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{

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