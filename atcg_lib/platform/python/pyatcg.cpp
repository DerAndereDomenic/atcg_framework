#include "bindings/Bindings.h"

// The entry point defines non-inline functions, so it may only be included by this translation unit
#ifdef ATCG_PYTHON_MODULE
    #include <Core/EntryPoint.h>
#endif

#define STRINGIFY(x)       #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace pybind11::literals;

#ifdef ATCG_PYTHON_MODULE
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
        atcg::print_statistics();
        atcg::SystemRegistry::shutdown();
    }

private:
    atcg::ref_ptr<spdlog::logger> _logger;
};

//* This function isn't called but is needed for the linker
atcg::Application* atcg::createApplication()
{
    return nullptr;
}
#endif

inline void defineBindings(py::module_& m)
{
    m.doc() = R"pbdoc(
        Pybind11 atcg plugin
        -----------------------
        .. currentmodule:: pyatcg
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    atcg::python::BindingContext ctx(m);

    // Phase 1: declare all classes, enums and submodules. Base classes (Event, Asset, Material, Integrator) have to be
    // declared before their subclasses.
    atcg::python::bindCore(ctx);
    atcg::python::bindMath(ctx);
    atcg::python::bindAsset(ctx);
    atcg::python::bindDataStructure(ctx);
    atcg::python::bindRenderer(ctx);
    atcg::python::bindUtils(ctx);
    atcg::python::bindMaterial(ctx);
    atcg::python::bindComponents(ctx);
    atcg::python::bindScene(ctx);
    atcg::python::bindGUI(ctx);
    atcg::python::bindNetwork(ctx);
    atcg::python::bindScripting(ctx);
    atcg::python::bindPlugin(ctx);
#ifdef ATCG_CUDA_BACKEND
    atcg::python::bindOptix(ctx);
#endif
#ifndef ATCG_HEADLESS
    atcg::python::bindImGui(ctx);
#endif

// On module initialization and destruction
#ifdef ATCG_PYTHON_MODULE
    py::class_<PythonContext>(m, "PythonContext").def(py::init<>());
    static PythonContext context;

    py::module atexit = py::module::import("atexit");
    atexit.attr("register")(py::cpp_function([]() { context.onExit(); }));
#endif

    // Phase 2: define the members of all declared types
    ctx.defineAll();

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    defineBindings(m);
}
