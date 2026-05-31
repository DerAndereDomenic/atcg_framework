#include <stdio.h>
#include <Plugin/Plugin.h>

#include <ATCG.h>

class TestPlugin : public atcg::DummyPluginBase
{
public:
    ATCG_PLUGIN_CLASS(TestPlugin);

    static atcg::ref_ptr<atcg::DummyPluginBase> create() { return atcg::make_ref<TestPlugin>(); }

    virtual void doStuff() override
    {
        ImGui::Begin("Test Window Plugin");
        ImGui::End();
    }
};

extern "C" __declspec(dllexport) void registerPlugin(atcg::PluginRegistry& registry)
{
    registry.registerClass<atcg::DummyPluginBase>("TestPlugin", &TestPlugin::create);
}