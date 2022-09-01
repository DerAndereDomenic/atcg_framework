#include <Core/Application.h>
#include <Core/API.h>

namespace atcg
{
    Application::Application()
    {
        _window = std::make_unique<Window>(WindowProps());
        _window->setEventCallback(ATCG_BIND_EVENT_FN(Application::onEvent));
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
        while(_running)
        {
            for(Layer* layer : _layer_stack)
            {
                layer->onUpdate(0);
            }

            //First finish the main content of all layers before doing any imgui stuff
            for(Layer* layer : _layer_stack)
            {
                layer->onImGuiRender();
            }

            _window->onUpdate();
        }
    }

    bool Application::onWindowClose(WindowCloseEvent& e)
    {
        _running = false;
        return true;
    }

    bool Application::onWindowResize(WindowResizeEvent& e)
    {
        return false;
    }
}