#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <ImGuizmo.h>

#include <random>

class MandelbulbLayer : public atcg::Layer
{
public:
    MandelbulbLayer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Application::get()->enableDockSpace(false);
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller  = atcg::make_ref<atcg::FirstPersonController>(aspect_ratio);
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();
    }

    virtual void onImGuiRender() override {}

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override
    {
        camera_controller->onEvent(event);

        atcg::EventDispatcher dispatcher(event);
        dispatcher.dispatch<atcg::ViewportResizeEvent>(ATCG_BIND_EVENT_FN(MandelbulbLayer::onViewportResized));
    }

    bool onViewportResized(atcg::ViewportResizeEvent* event)
    {
        atcg::WindowResizeEvent resize_event(event->getWidth(), event->getHeight());
        camera_controller->onEvent(&resize_event);
        return false;
    }


private:
    atcg::ref_ptr<atcg::FirstPersonController> camera_controller;
};

class Mandelbulb : public atcg::Application
{
public:
    Mandelbulb() : atcg::Application() { pushLayer(new MandelbulbLayer("Layer")); }

    ~Mandelbulb() {}
};

atcg::Application* atcg::createApplication()
{
    return new Mandelbulb;
}