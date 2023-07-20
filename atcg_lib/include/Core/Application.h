#pragma once

#include <Core/LayerStack.h>
#include <Core/Window.h>

#include <Events/WindowEvent.h>

#include <ImGui/ImGuiLayer.h>

#include <Core/Memory.h>

namespace atcg
{
class Application;
int atcg_main(Application* app);
}    // namespace atcg
int python_main(atcg::Layer* layer);

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
     * @brief Construct a new Application object
     *
     * @param props The window properties
     */
    Application(const WindowProps& props);

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
    void onEvent(Event* e);

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

    /**
     * @brief Get the Window object
     *
     * @return const atcg::scope_ptr<Window>& The window
     */
    inline const atcg::scope_ptr<Window>& getWindow() const { return _window; }

    /**
     * @brief Get an instance of the application
     *
     * @return Application* The application instance
     */
    inline static Application* get() { return s_instance; }

private:
    void run();
    bool onWindowClose(WindowCloseEvent* e);
    bool onWindowResize(WindowResizeEvent* e);
    void init(const WindowProps& props);

private:
    bool _running = false;
    atcg::scope_ptr<Window> _window;
    ImGuiLayer* _imgui_layer;
    LayerStack _layer_stack;

    friend int atcg::atcg_main(Application* app);
    friend int ::python_main(atcg::Layer* layer);    // Entry point for python bindings
    static Application* s_instance;
};

// Entry point for client
Application* createApplication();
}    // namespace atcg