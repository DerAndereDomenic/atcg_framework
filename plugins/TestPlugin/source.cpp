#include <stdio.h>
#include <Plugin/Plugin.h>

#include <ATCG.h>

class TestPlugin : public atcg::DummyPluginBase
{
public:
    ATCG_PLUGIN_CLASS(TestPlugin);

    static atcg::ref_ptr<atcg::DummyPluginBase> create() { return atcg::make_ref<TestPlugin>(); }

    virtual void doStuff(const atcg::ref_ptr<atcg::Scene>& scene, const atcg::ref_ptr<atcg::Camera>& camera) override
    {
        // atcg::SceneRenderer::render(scene, camera, atcg::Renderer::getFramebuffer());
    }
};

ATCG_PLUGIN_LIBRARY();

extern "C" __declspec(dllexport) void registerPlugin(atcg::PluginRegistry& registry)
{
    registry.registerClass<atcg::DummyPluginBase>("TestPlugin", &TestPlugin::create);
}