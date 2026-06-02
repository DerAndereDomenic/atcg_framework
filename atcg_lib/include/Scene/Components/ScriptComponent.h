#pragma once

#include <Asset/AssetManagerSystem.h>
#include <Scripting/Behavior.h>
#include <Scripting/Script.h>
#include <Scene/ComponentGUIHandler.h>
#include <Scene/ComponentSerializer.h>

namespace atcg
{
struct ScriptComponent
{
    ScriptComponent() = default;

    ScriptComponent(const atcg::ref_ptr<Script>& script) { setScript(script); }

    ATCG_INLINE atcg::ref_ptr<Script> script() const { return AssetManager::getAsset<Script>(script_handle); }

    ATCG_INLINE void setScript(const atcg::ref_ptr<Script>& script)
    {
        if(AssetManager::isAssetHandleValid(script->handle))
        {
            script_handle = script->handle;
        }
        else
        {
            auto script_path = script->getFilePath();
            // Get the filename without extension as name
            auto script_name = script_path.stem().string();
            script_handle    = AssetManager::registerAsset(script, script_name);
        }
        _behavior = nullptr;
    }

    ATCG_INLINE atcg::ref_ptr<Behavior>
    behavior(const atcg::ref_ptr<Scene>& scene, atcg::Entity entity, bool recreate = false)
    {
        auto scr = script();

        if(!scr) return nullptr;

        if(recreate || !_behavior)
        {
            ATCG_DEBUG("Create new behavior");
            _behavior = scr->createBehavior(scene, entity);
        }

        return _behavior;
    }

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Script"; }

    AssetHandle script_handle = 0;

private:
    atcg::ref_ptr<Behavior> _behavior = nullptr;
};

namespace Serialization
{
ATCG_DECLARE_COMPONENT_SERIALIZER(ScriptComponent);
}

namespace GUI
{
ATCG_DECLARE_COMPONENT_GUI_RENDERER(ScriptComponent);
}
}    // namespace atcg