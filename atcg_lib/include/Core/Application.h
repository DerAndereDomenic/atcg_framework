#pragma once

#include <Core/LayerStack.h>
#include <Core/Window.h>

#include <Events/WindowEvent.h>

#include <memory>

int main(int argc, char** argv);

namespace atcg
{
    /**
     * @brief Each exercise is an application
     * 
     */
    class Application
    {
    public:
        /**
         * @brief Construct a new Application object
         * 
         */
        Application();

        /**
         * @brief Destroy the Application object
         * 
         */
        virtual ~Application();

        /**
         * @brief Handles events
         * 
         * @param e The event
         */
        void onEvent(Event& e);

        /**
         * @brief Push a layer to the application
         * 
         * @param layer The layer
         */
        void pushLayer(Layer* layer);

        /**
         * @brief Close the application
         * 
         */
        void close();

    private:
        void run();
        bool onWindowClose(WindowCloseEvent& e);
        bool onWindowResize(WindowResizeEvent& e);
    private:
        bool _running = false;
        LayerStack _layer_stack;
        std::unique_ptr<Window> _window;

        friend int ::main(int argc, char** argv);
    };

    // Entry point for client
    Application* createApplication();
}