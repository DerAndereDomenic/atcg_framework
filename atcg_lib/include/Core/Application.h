#pragma once

#include <Core/Memory.h>
#include <Core/LayerStack.h>
#include <Core/Window.h>
#include <Events/WindowEvent.h>
#include <Events/KeyEvent.h>
#include <Core/Platform.h>
#include <Scene/RevisionStack.h>
#include <Renderer/ShaderManager.h>
#include <Renderer/Renderer.h>
#include <Renderer/VRSystem.h>
#include <Renderer/ContextManager.h>
#include <Scripting/ScriptEngine.h>

#ifndef ATCG_HEADLESS
    #include <ImGui/ImGuiLayer.h>
#endif

#ifdef ATCG_ENABLE_OPTIX
    #include <Core/RaytracingContextManager.h>
#endif
namespace atcg
{
class Application;
int atcg_main();
}    // namespace atcg

namespace atcg
{
// Based on Hazel Engine (https://github.com/TheCherno/Hazel)
// Modified by Domenic Zingsheim in 2022

/**
 * @brief A class to model an application
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
    ATCG_INLINE void enableDockSpace(bool enable)
    {
#ifndef ATCG_HEADLESS
        _imgui_layer->enableDockSpace(enable);
#endif
    }

#ifndef ATCG_HEADLESS
    /**
     * @brief Get the imgui layer.
     *
     * @return The imgui layer
     */
    ATCG_INLINE ImGuiLayer* getImGuiLayer() const { return _imgui_layer; }
#endif
    /**
     * @brief Run the application
     */
    virtual void run();

private:
    bool onWindowClose(WindowCloseEvent* e);
    bool onWindowResize(WindowResizeEvent* e);
    bool onViewportResize(ViewportResizeEvent* e);
    bool onKeyPress(KeyPressedEvent* e);
    void init(const WindowProps& props);

private:
    bool _running = false;
    atcg::ref_ptr<ContextManagerSystem> _context_manager;
#ifdef ATCG_ENABLE_OPTIX
    atcg::ref_ptr<RaytracingContextManagerSystem> _rt_context_manager;
#endif
    atcg::scope_ptr<Window> _window;
#ifndef ATCG_HEADLESS
    ImGuiLayer* _imgui_layer;
#endif
    LayerStack _layer_stack;

    // Systems
    atcg::ref_ptr<ShaderManagerSystem> _shader_manager;
    atcg::ref_ptr<RendererSystem> _renderer;
    atcg::ref_ptr<VRSystem> _vr_system;
    atcg::ref_ptr<ScriptEngine> _script_engine;
    atcg::ref_ptr<RevisionSystem> _revision_system;

    friend int atcg::atcg_main();
    static Application* s_instance;
};

// Entry point for client
Application* createApplication();
}    // namespace atcg