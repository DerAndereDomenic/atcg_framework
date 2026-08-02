#include <Core/Application.h>

#include <Core/SystemRegistry.h>
#include <Core/Assert.h>
#include <Core/KeyCodes.h>
#include <Renderer/Renderer.h>
#include <Renderer/VRSystem.h>
#include <Renderer/ShaderManager.h>
#include <Scene/RevisionStack.h>
#include <Asset/Project.h>


namespace atcg
{
Application* Application::s_instance = nullptr;

Application::Application()
{
    init(WindowProps());
}

Application::Application(const WindowProps& props)
{
    init(props);
}

Application::~Application()
{
    _layer_stack.clear();
    _systems->shutdownSystems();
}

void Application::init(const WindowProps& props)
{
    ATCG_ASSERT(!s_instance, "There can only be one application instance at a time.");
    ATCG_ASSERT(SystemRegistry::instance(), "SystemRegistry must be initialized before initializing the Application");

    _systems = atcg::make_scope<SystemCollection>();
    _systems->initSystems(props, ATCG_BIND_EVENT_FN(Application::onEvent));

    // Create an active project
    atcg::Project::create("./DefaultProject");

    s_instance = this;

#ifndef ATCG_HEADLESS
    _imgui_layer = new ImGuiLayer();
    _layer_stack.pushOverlay(_imgui_layer);
    _imgui_layer->onAttach();
#endif
}

void Application::pushLayer(Layer* layer)
{
    ATCG_ASSERT(layer, "Layer cannot be nullptr");

    _layer_stack.pushLayer(layer);
    layer->onAttach();
}

void Application::close()
{
    ATCG_ASSERT(!_running, "Can only close running applications");

    _running = false;
}

void Application::onEvent(Event* e)
{
    EventDispatcher dispatcher(e);
    dispatcher.dispatch<WindowCloseEvent>(ATCG_BIND_EVENT_FN(Application::onWindowClose));
    dispatcher.dispatch<WindowResizeEvent>(ATCG_BIND_EVENT_FN(Application::onWindowResize));
    dispatcher.dispatch<ViewportResizeEvent>(ATCG_BIND_EVENT_FN(Application::onViewportResize));
    dispatcher.dispatch<KeyPressedEvent>(ATCG_BIND_EVENT_FN(Application::onKeyPress));

    for(auto it = _layer_stack.rbegin(); it != _layer_stack.rend(); ++it)
    {
        if(e->handled) break;
        (*it)->onEvent(e);
    }
}

glm::ivec2 Application::getViewportSize() const
{
#ifndef ATCG_HEADLESS
    if(_imgui_layer->dockspaceEnabled())
    {
        return _imgui_layer->getViewportSize();
    }
#endif

    return glm::ivec2(getWindow()->getWidth(), getWindow()->getHeight());
}

glm::ivec2 Application::getViewportPosition() const
{
#ifndef ATCG_HEADLESS
    if(_imgui_layer->dockspaceEnabled())
    {
        return _imgui_layer->getViewportPosition();
    }
#endif

    return glm::ivec2(0);
}

void Application::run()
{
    ATCG_ASSERT(!_running, "Can only start application once");
    ATCG_ASSERT(SystemRegistry::instance(), "System registry must be initalized before running the app");
    ATCG_ASSERT(SystemRegistry::instance()->hasSystem<RendererSystem>(),
                "There must be a registered Renderer before starting the app");
    ATCG_ASSERT(SystemRegistry::instance()->hasSystem<ShaderManagerSystem>(),
                "There must be a registered ShaderManager before starting the app");
    ATCG_ASSERT(SystemRegistry::instance()->hasSystem<VRSystem>(),
                "There must be a registered VRSystem before starting the app");
    ATCG_ASSERT(SystemRegistry::instance()->hasSystem<spdlog::logger>(),
                "There must be a registered Logger before starting the app");

    _running          = true;
    auto last_time    = std::chrono::high_resolution_clock::now();
    auto current_time = std::chrono::high_resolution_clock::now();
    float delta_time  = 1.0f / 60.0f;    // Only for first frame
    float total_time  = 0.0f;
    while(_running)
    {
        last_time = current_time;

        VR::doTracking();

        for(Layer* layer: _layer_stack)
        {
            layer->onUpdate(delta_time);
        }
        Renderer::finishFrame();

// First finish the main content of all layers before doing any imgui stuff
#ifndef ATCG_HEADLESS
        _imgui_layer->begin();
        for(Layer* layer: _layer_stack)
        {
            layer->onImGuiRender();
        }
        _imgui_layer->end();
#endif

        VR::onUpdate(delta_time);
        VR::emitEvents();
        getWindow()->onUpdate();
        ++_application_counter;

#ifndef ATCG_HEADLESS
        glm::ivec2 viewport_size = _imgui_layer->getViewportSize();
        if(_imgui_layer->dockspaceEnabled() && (viewport_size.x != Renderer::getFramebuffer()->width() ||
                                                viewport_size.y != Renderer::getFramebuffer()->height()))
        {
            ViewportResizeEvent event(viewport_size.x, viewport_size.y);
            onEvent(&event);
        }
#endif

        current_time = std::chrono::high_resolution_clock::now();


        // Only check for shader reloading every second
        if(total_time >= 1.0f)
        {
            total_time = 0.0f;
            ShaderManager::onUpdate();
        }

        delta_time =
            std::chrono::duration_cast<std::chrono::microseconds>(current_time - last_time).count() / 1000000.0f;

        total_time += delta_time;
    }
}

bool Application::onWindowClose(WindowCloseEvent* e)
{
    _running = false;
    return true;
}

bool Application::onWindowResize(WindowResizeEvent* e)
{
    Renderer::resize(e->getWidth(), e->getHeight());
    return false;
}

bool Application::onViewportResize(ViewportResizeEvent* e)
{
    Renderer::resize(e->getWidth(), e->getHeight());
    return false;
}

bool Application::onKeyPress(KeyPressedEvent* e)
{
    if(e->getKeyCode() == ATCG_KEY_F11)
    {
        getWindow()->toggleFullscreen();
    }

    if(e->getKeyCode() == ATCG_KEY_Y && atcg::Input::isKeyPressed(ATCG_KEY_LEFT_CONTROL))
    {
        RevisionStack::rollback();
    }

    if(e->getKeyCode() == ATCG_KEY_Z && atcg::Input::isKeyPressed(ATCG_KEY_LEFT_CONTROL))
    {
        RevisionStack::apply();
    }

    return false;
}
}    // namespace atcg