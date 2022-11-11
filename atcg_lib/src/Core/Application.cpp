#include <Core/Application.h>
#include <Core/API.h>
#include <Renderer/Renderer.h>
#include <Renderer/ShaderManager.h>
#include <chrono>

namespace atcg
{
    Application* Application::s_instance = nullptr;

    Application::Application()
    {
        _window = std::make_unique<Window>(WindowProps());
        _window->setEventCallback(ATCG_BIND_EVENT_FN(Application::onEvent));

        Renderer::init(_window->getWidth(), _window->getHeight());

        Renderer::setClearColor(glm::vec4(76.0f, 76.0f, 128.0f, 255.0f) / 255.0f);

        ShaderManager::addShaderFromName("base");
        ShaderManager::addShaderFromName("flat");
        ShaderManager::addShaderFromName("edge");
        ShaderManager::addShaderFromName("circle");
        ShaderManager::addShaderFromName("grid");

        s_instance = this;

        _imgui_layer = new ImGuiLayer();
        _layer_stack.pushOverlay(_imgui_layer);
        _imgui_layer->onAttach();
    }

    Application::~Application()
    {
        
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

    void Application::onEvent(Event& e)
    {
        EventDispatcher dispatcher(e);
        dispatcher.dispatch<WindowCloseEvent>(ATCG_BIND_EVENT_FN(Application::onWindowClose));
        dispatcher.dispatch<WindowResizeEvent>(ATCG_BIND_EVENT_FN(Application::onWindowResize));
        
        for(auto it = _layer_stack.rbegin(); it != _layer_stack.rend(); ++it)
        {
            if(e.handled)
                break;
            (*it)->onEvent(e);
        }
    }

    void Application::run()
    {
        _running = true;
        auto last_time = std::chrono::high_resolution_clock::now();
        auto current_time = std::chrono::high_resolution_clock::now();
        float delta_time = 1.0f/60.0f; //Only for first frame
        float total_time = 0.0f;
        while(_running)
        {
            last_time = current_time;

            Renderer::useScreenBuffer();
            for(Layer* layer : _layer_stack)
            {
                layer->onUpdate(delta_time);
            }
            Renderer::finishFrame();

            //First finish the main content of all layers before doing any imgui stuff
            _imgui_layer->begin();
            for(Layer* layer : _layer_stack)
            {
                layer->onImGuiRender();
            }
            _imgui_layer->end();

            _window->onUpdate();

            current_time = std::chrono::high_resolution_clock::now();

            delta_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time).count() / 1000.0f;

            total_time += delta_time;

            //Only check for shader reloading every second
            if(total_time >= 1.0f)
            {
                total_time = 0.0f;
                ShaderManager::onUpdate();
            }
        }
    }

    bool Application::onWindowClose(WindowCloseEvent& e)
    {
        _running = false;
        return true;
    }

    bool Application::onWindowResize(WindowResizeEvent& e)
    {
        Renderer::setViewport(0, 0, e.getWidth(), e.getHeight());
        return false;
    }
}