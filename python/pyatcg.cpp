#include <pybind11/pybind11.h>
#include <ATCG.h>

#define STRINGIFY(x)       #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

class PythonLayer : public atcg::Layer
{
public:
    PythonLayer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override {}

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        // Renders points and curves
        atcg::Renderer::clear();
    }

    virtual void onImGuiRender() override {}

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event& event) override {}

private:
};

class PythonApplication : public atcg::Application
{
public:
    PythonApplication() : atcg::Application() { pushLayer(new PythonLayer("Layer")); }

    ~PythonApplication() {}
};

atcg::Application* atcg::createApplication()
{
    return new PythonApplication;
}

int entry_point()
{
    atcg::Application* app = atcg::createApplication();
    app->run();

    delete app;

    return 0;
}

namespace py = pybind11;

PYBIND11_MODULE(pyatcg, m)
{
    m.doc() = R"pbdoc(
        Pybind11 atcg plugin
        -----------------------
        .. currentmodule:: pyatcg
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    m.def("start", &entry_point, "Start the application");


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}