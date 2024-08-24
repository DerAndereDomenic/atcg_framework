#pragma once

#include <Core/Memory.h>
#include <Core/LayerStack.h>
#include <Core/Window.h>
#include <Events/WindowEvent.h>
#include <ImGui/ImGuiLayer.h>
#include <Core/Platform.h>
#include <Renderer/ShaderManager.h>

namespace atcg
{
class Application;
int atcg_main(Application* app);
}    // namespace atcg
int python_main(atcg::Application* app);

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
     * @brief Get the viewport size (without title bar).
     * If docking is not enabled, this is the same as window->width() and window->height()
     *
     * @return The viewport size
     */
    glm::ivec2 getViewportSize() const;

    /**
     * @brief Get the viewport posititon (without title bar).
     * If docking is not enabled, this is (0,0)
     *
     * @return The viewport position
     */
    glm::ivec2 getViewportPosition() const;

    /**
     * @brief Get the Window object
     *
     * @return const atcg::scope_ptr<Window>& The window
     */
    ATCG_INLINE const atcg::scope_ptr<Window>& getWindow() const { return _window; }

    /**
     * @brief Get an instance of the application
     *
     * @return Application* The application instance
     */
    ATCG_INLINE static Application* get() { return s_instance; }

    /**
     * @brief Enable or disable Dock spaces
     *
     * @param enable If dockspaces should be enabled
     */
    ATCG_INLINE void enableDockSpace(bool enable) { _imgui_layer->enableDockSpace(enable); }

    /**
     * @brief Get the imgui layer.
     *
     * @return The imgui layer
     */
    ATCG_INLINE ImGuiLayer* getImGuiLayer() const { return _imgui_layer; }

protected:
    virtual void run();

private:
    bool onWindowClose(WindowCloseEvent* e);
    bool onWindowResize(WindowResizeEvent* e);
    bool onViewportResize(ViewportResizeEvent* e);
    void init(const WindowProps& props);

private:
    bool _running = false;
    atcg::scope_ptr<Window> _window;
    ImGuiLayer* _imgui_layer;
    LayerStack _layer_stack;

    // Systems
    atcg::ref_ptr<Logger> _logger;
    atcg::ref_ptr<ShaderManager> _shader_manager;

    friend int atcg::atcg_main(Application* app);
    friend int ::python_main(atcg::Application* app);    // Entry point for python bindings
    static Application* s_instance;
};

// Entry point for client
Application* createApplication();
}    // namespace atcg