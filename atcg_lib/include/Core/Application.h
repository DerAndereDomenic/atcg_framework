#pragma once

#include <Core/LayerStack.h>

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

    private:
        bool _running = false;
        LayerStack _layer_stack;
    };

    // Entry point for client
    Application* createApplication();
}