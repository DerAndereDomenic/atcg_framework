#include <Core/Application.h>

#include <Renderer/Renderer.h>
#include <Renderer/VRRenderer.h>
#include <Renderer/ShaderManager.h>


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

Application::~Application() {}

void Application::init(const WindowProps& props)
{
    Log::init();

    _window = atcg::make_scope<Window>(props);
    _window->setEventCallback(ATCG_BIND_EVENT_FN(Application::onEvent));

    Renderer::init(_window->getWidth(), _window->getHeight());
    VRRenderer::init(ATCG_BIND_EVENT_FN(Application::onEvent));

    Renderer::setClearColor(glm::vec4(76.0f, 76.0f, 128.0f, 255.0f) / 255.0f);

    s_instance = this;

    _imgui_layer = new ImGuiLayer();
    _layer_stack.pushOverlay(_imgui_layer);
    _imgui_layer->onAttach();
}

void Application::pushLayer(Layer* layer)
{
    _layer_stack.pushLayer(layer);
    layer->onAttach();
}

void Application::close()
{
    _running = false;
}

void Application::onEvent(Event* e)
{
    EventDispatcher dispatcher(e);
    dispatcher.dispatch<WindowCloseEvent>(ATCG_BIND_EVENT_FN(Application::onWindowClose));
    dispatcher.dispatch<WindowResizeEvent>(ATCG_BIND_EVENT_FN(Application::onWindowResize));
    dispatcher.dispatch<ViewportResizeEvent>(ATCG_BIND_EVENT_FN(Application::onViewportResize));

    for(auto it = _layer_stack.rbegin(); it != _layer_stack.rend(); ++it)
    {
        if(e->handled) break;
        (*it)->onEvent(e);
    }
}

glm::ivec2 Application::getViewportSize() const
{
    if(_imgui_layer->dockspaceEnabled())
    {
        return _imgui_layer->getViewportSize();
    }

    return glm::ivec2(_window->getWidth(), _window->getHeight());
}

glm::ivec2 Application::getViewportPosition() const
{
    if(_imgui_layer->dockspaceEnabled())
    {
        return _imgui_layer->getViewportPosition();
    }

    return glm::ivec2(0);
}

void Application::run()
{
    _running          = true;
    auto last_time    = std::chrono::high_resolution_clock::now();
    auto current_time = std::chrono::high_resolution_clock::now();
    float delta_time  = 1.0f / 60.0f;    // Only for first frame
    float total_time  = 0.0f;
    while(_running)
    {
        last_time = current_time;

        Renderer::useScreenBuffer();
        for(Layer* layer: _layer_stack)
        {
            layer->onUpdate(delta_time);
        }
        Renderer::finishFrame();

        // First finish the main content of all layers before doing any imgui stuff
        _imgui_layer->begin();
        for(Layer* layer: _layer_stack)
        {
            layer->onImGuiRender();
        }
        _imgui_layer->end();

        VRRenderer::onUpdate(delta_time);
        _window->onUpdate();
        glm::ivec2 viewport_size = _imgui_layer->getViewportSize();
        if(_imgui_layer->dockspaceEnabled() && (viewport_size.x != Renderer::getFramebuffer()->width() ||
                                                viewport_size.y != Renderer::getFramebuffer()->height()))
        {
            ViewportResizeEvent event(viewport_size.x, viewport_size.y);
            onEvent(&event);
        }

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
}    // namespace atcg