#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <ATCG.h>

#define STRINGIFY(x)       #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


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

    ~PythonApplication() {}
};

atcg::Application* atcg::createApplication()
{
    return new PythonApplication;
}

int entry_point(atcg::Layer* layer)
{
    atcg::Application* app = atcg::createApplication();
    app->pushLayer(layer);
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

    m.def("start", &entry_point, "Start the application.");
    m.def("width",
          []()
          {
              const auto& window = atcg::Application::get()->getWindow();
              return (float)window->getWidth();
          });

    m.def("height",
          []()
          {
              const auto& window = atcg::Application::get()->getWindow();
              return (float)window->getHeight();
          });

    py::class_<atcg::Layer, PythonLayer>(m, "Layer")
        .def(py::init<>())
        .def("onAttach", &atcg::Layer::onAttach)
        .def("onUpdate", &atcg::Layer::onUpdate)
        .def("onImGuiRender", &atcg::Layer::onImGuiRender)
        .def("onEvent", &atcg::Layer::onEvent);

    py::class_<atcg::Event>(m, "Event");

    py::class_<atcg::Renderer>(m, "Renderer")
        .def("setClearColor",
             [](const float r, const float g, const float b, const float a)
             { atcg::Renderer::setClearColor(glm::vec4(r, g, b, a)); })
        .def_static("clear", &atcg::Renderer::clear);

    py::class_<atcg::PerspectiveCamera, std::shared_ptr<atcg::PerspectiveCamera>>(m, "PerspectiveCamera");

    py::class_<atcg::CameraController>(m, "CameraController")
        .def(py::init<float>())
        .def("onUpdate", &atcg::CameraController::onUpdate)
        .def("onEvent", &atcg::CameraController::onEvent)
        .def("getCamera", &atcg::CameraController::getCamera);

    py::class_<atcg::Shader, std::shared_ptr<atcg::Shader>>(m, "Shader");

    py::class_<atcg::ShaderManager>(m, "ShaderManager").def_static("getShader", &atcg::ShaderManager::getShader);


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}