#include <Core/Application.h>
#include <Events/WindowEvent.h>

namespace atcg
{
    Application::Application()
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
        }
    }
}