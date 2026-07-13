#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <torch/python.h>
#ifdef ATCG_PYTHON_MODULE
    #include <Core/EntryPoint.h>
#endif
#include <ATCG.h>

class PythonLayer : public atcg::Layer
{
public:
    PythonLayer(const std::string& name = "Layer") : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override { PYBIND11_OVERRIDE(void, atcg::Layer, onAttach); }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override { PYBIND11_OVERRIDE(void, atcg::Layer, onUpdate, delta_time); }

    virtual void onImGuiRender() override { PYBIND11_OVERRIDE(void, atcg::Layer, onImGuiRender); }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override { PYBIND11_OVERRIDE(void, atcg::Layer, onEvent, event); }

private:
};

class PythonApplication : public atcg::Application
{
public:
    PythonApplication() : atcg::Application() {}

    PythonApplication(const atcg::WindowProps& props) : atcg::Application(props) {}

    PythonApplication(atcg::Layer* layer) : atcg::Application() { pushLayer(layer); }

    PythonApplication(atcg::Layer* layer, const atcg::WindowProps& props) : atcg::Application(props)
    {
        pushLayer(layer);
    }

    ~PythonApplication() {}
};

class PythonContext
{
public:
    PythonContext()
    {
        _logger = spdlog::stdout_color_mt("ATCG");
        _logger->set_pattern("%^[%T] %n: %v%$");
        _logger->set_level(spdlog::level::trace);
        atcg::SystemRegistry::init();
        atcg::SystemRegistry::instance()->registerSystem(_logger.get());
    }

    void onExit()
    {
#ifdef ATCG_PYTHON_MODULE
        atcg::print_statistics();
#endif
        atcg::SystemRegistry::shutdown();
    }

private:
    atcg::ref_ptr<spdlog::logger> _logger;
};


//* This function isn't called but is needed for the linker
#ifdef ATCG_PYTHON_MODULE
atcg::Application* atcg::createApplication()
{
    return nullptr;
}
#endif


PYBIND11_DECLARE_HOLDER_TYPE(T, atcg::ref_ptr<T>);