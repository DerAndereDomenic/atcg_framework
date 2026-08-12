#include <Core/SystemCollection.h>

#include <Core/SystemRegistry.h>

// Systems
#include <Plugin/Plugin.h>
#include <Renderer/ContextManager.h>
#include <Renderer/GraphicsAPI.h>
#include <Renderer/Renderer.h>
#include <Renderer/ShaderManager.h>
#include <Renderer/VRSystem.h>
#include <Renderer/RenderPassRegistry.h>
#include <Scene/ComponentRegistry.h>
#include <Scene/RevisionStack.h>
#include <Scene/SceneRenderer.h>
#include <Scripting/ScriptEngine.h>

#include <Material/Material.h>
#include <Material/Medium.h>
#include <Material/MaterialRegistry.h>
#include <Material/MediumRegistry.h>
#include <Material/DielectricMaterial.h>
#include <Material/OpaqueMaterial.h>
#include <Material/NullMaterial.h>
#include <Material/HomogeneousMedium.h>
#include <Material/HeterogeneousMedium.h>

// Render Passes
#include <Renderer/RenderPasses/TonemapPass.h>
#include <Renderer/RenderPasses/DepthPass.h>
#include <Renderer/RenderPasses/ForwardPass.h>
#include <Renderer/RenderPasses/ShadowPass.h>
#include <Renderer/RenderPasses/OutlinePass.h>
#include <Renderer/RenderPasses/BlitPass.h>
#include <Renderer/RenderPasses/OutputPass.h>

// Renderer
#include <Core/Window.h>
#include <Renderer/Renderer.h>
#include <Renderer/VRSystem.h>
#include <Renderer/ShaderManager.h>

// Optix Components
#ifdef ATCG_CUDA_BACKEND
    #include <Renderer/RaytracingContextManager.h>
    #include <BSDF/BSDFRegistry.h>
    #include <Integrator/IntegratorRegistry.h>
    #include <BSDF/PBRBSDF.h>
    #include <BSDF/NullBSDF.h>
    #include <BSDF/DielectricBSDF.h>
    #include <Integrator/PathtracingIntegrator.h>
    #include <Integrator/VolPathtracingIntegrator.h>
    #include <Integrator/PhotonMapIntegrator.h>
#endif

namespace atcg
{

class SystemCollection::Impl
{
public:
    Impl();

    ~Impl();

    void initSystems(const WindowProps& props, const Window::EventCallbackFn& event_callback);

    void shutdownSystems();

    atcg::ref_ptr<ContextManagerSystem> _context_manager;
#ifdef ATCG_CUDA_BACKEND
    atcg::ref_ptr<RaytracingContextManagerSystem> _rt_context_manager;
    atcg::ref_ptr<BSDFRegistry::Registry> _bsdf_registry;
    atcg::ref_ptr<IntegratorRegistry::Registry> _integrator_registry;
#endif
    atcg::scope_ptr<Window> _window;
    atcg::ref_ptr<AssetManagerSystem> _asset_manager;
    atcg::ref_ptr<ShaderManagerSystem> _shader_manager;
    atcg::ref_ptr<MaterialRegistry::Registry> _material_registry;
    atcg::ref_ptr<MediumRegistry::Registry> _medium_registry;
    atcg::ref_ptr<RenderPassRegistry::Registry> _render_pass_registry;
    atcg::ref_ptr<SceneRendererSystem> _scene_renderer;
    atcg::ref_ptr<RendererSystem> _renderer;
    atcg::ref_ptr<VRSystem> _vr_system;
    atcg::ref_ptr<ScriptEngine> _script_engine;
    atcg::ref_ptr<RevisionSystem> _revision_system;
    atcg::ref_ptr<GraphicsAPI> _graphics_api;
    atcg::ref_ptr<ComponentRegistrySystem> _component_registry;
    atcg::ref_ptr<PluginManagerSystem> _plugin_manager;
};

SystemCollection::Impl::Impl() {}

SystemCollection::Impl::~Impl() {}

void SystemCollection::Impl::initSystems(const WindowProps& props, const Window::EventCallbackFn& event_callback)
{
    _asset_manager = atcg::make_ref<AssetManagerSystem>();
    SystemRegistry::instance()->registerSystem(_asset_manager.get());

    _context_manager = atcg::make_ref<ContextManagerSystem>();
    SystemRegistry::instance()->registerSystem(_context_manager.get());

#ifdef ATCG_CUDA_BACKEND
    atcg::RaytracingContext::initRaytracingAPI();
    _rt_context_manager = atcg::make_ref<RaytracingContextManagerSystem>();
    SystemRegistry::instance()->registerSystem(_rt_context_manager.get());

    _bsdf_registry = atcg::make_ref<BSDFRegistry::Registry>();
    PBRBSDF::registerBSDF(_bsdf_registry.get());
    DielectricBSDF::registerBSDF(_bsdf_registry.get());
    NullBSDF::registerBSDF(_bsdf_registry.get());
    SystemRegistry::instance()->registerSystem(_bsdf_registry.get());

    _integrator_registry = atcg::make_ref<IntegratorRegistry::Registry>();
    VolPathtracingIntegrator::registerIntegrator(_integrator_registry.get());
    PathtracingIntegrator::registerIntegrator(_integrator_registry.get());
    PhotonMapIntegrator::registerIntegrator(_integrator_registry.get());
    SystemRegistry::instance()->registerSystem(_integrator_registry.get());
#endif

    _shader_manager = atcg::make_ref<ShaderManagerSystem>();
    SystemRegistry::instance()->registerSystem(_shader_manager.get());

    _window = atcg::make_scope<Window>(props);
    _window->setEventCallback(event_callback);

    _graphics_api = atcg::make_ref<GraphicsAPI>();
    _graphics_api->init();
    SystemRegistry::instance()->registerSystem(_graphics_api.get());

    _renderer = atcg::make_ref<RendererSystem>();
    _renderer->init(_window->getWidth(), _window->getHeight(), _window->getContext(), _shader_manager);
    SystemRegistry::instance()->registerSystem(_renderer.get());

    // Needs to be called after the renderer is initialized
    _asset_manager->loadStandardAssets();

    _vr_system = atcg::make_ref<VRSystem>();
    _vr_system->init(event_callback);
    SystemRegistry::instance()->registerSystem(_vr_system.get());

    _revision_system = atcg::make_ref<RevisionSystem>();
    SystemRegistry::instance()->registerSystem(_revision_system.get());

    _component_registry = atcg::make_ref<ComponentRegistrySystem>();
    SystemRegistry::instance()->registerSystem(_component_registry.get());

    _script_engine = atcg::make_ref<PythonScriptEngine>();
    _script_engine->init();
    SystemRegistry::instance()->registerSystem(_script_engine.get());

    // Register the material types
    _material_registry = atcg::make_ref<MaterialRegistry::Registry>();
    OpaqueMaterial::registerMaterial(_material_registry.get());
    DielectricMaterial::registerMaterial(_material_registry.get());
    NullMaterial::registerMaterial(_material_registry.get());

    SystemRegistry::instance()->registerSystem(_material_registry.get());

    _medium_registry = atcg::make_ref<MediumRegistry::Registry>();
    HomogeneousMedium::registerMedium(_medium_registry.get());
    HeterogeneousMedium::registerMedium(_medium_registry.get());

    SystemRegistry::instance()->registerSystem(_medium_registry.get());

    _render_pass_registry = atcg::make_ref<RenderPassRegistry::Registry>();
    OutputPass::registerRenderPass(_render_pass_registry.get());
    TonemapPass::registerRenderPass(_render_pass_registry.get());
    DepthPass::registerRenderPass(_render_pass_registry.get());
    ForwardPass::registerRenderPass(_render_pass_registry.get());
    ShadowPass::registerRenderPass(_render_pass_registry.get());
    OutlinePass::registerRenderPass(_render_pass_registry.get());
    BlitPass::registerRenderPass(_render_pass_registry.get());

    SystemRegistry::instance()->registerSystem(_render_pass_registry.get());

    _scene_renderer = atcg::make_ref<SceneRendererSystem>(_renderer.get());
    SystemRegistry::instance()->registerSystem(_scene_renderer.get());

    _plugin_manager = atcg::make_ref<PluginManagerSystem>();
    SystemRegistry::instance()->registerSystem(_plugin_manager.get());
}

void SystemCollection::Impl::shutdownSystems()
{
    _revision_system->clearChache();
    if(_asset_manager) _asset_manager->destroy();
    if(_script_engine) _script_engine->destroy();
    if(_plugin_manager) _plugin_manager->releaseAllPlugins();
}

SystemCollection::SystemCollection()
{
    impl = std::make_unique<Impl>();
}

SystemCollection::~SystemCollection() {}

void SystemCollection::initSystems(const WindowProps& props, const Window::EventCallbackFn& event_callback)
{
    impl->initSystems(props, event_callback);
}

void SystemCollection::shutdownSystems()
{
    impl->shutdownSystems();
}

const atcg::scope_ptr<Window>& SystemCollection::getWindow() const
{
    return impl->_window;
}
}    // namespace atcg