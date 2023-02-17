#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <ATCG.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

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

    m.def("setSize",
          [](uint32_t width, uint32_t height)
          {
              const auto& window = atcg::Application::get()->getWindow();
              window->resize(width, height);
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
        .def_static("clear", &atcg::Renderer::clear)
        .def("renderMesh",
             [](const std::shared_ptr<atcg::Mesh>& mesh,
                const std::shared_ptr<atcg::Shader>& shader,
                const std::shared_ptr<atcg::PerspectiveCamera>& camera)
             { atcg::Renderer::draw(mesh, shader, camera); });

    py::class_<glm::vec3>(m, "Vector3", py::buffer_protocol())
        .def(py::init(
            [](py::buffer b)
            {
                py::buffer_info info = b.request();

                // Copy for now, is there a better method?
                glm::vec3 v;
                if(info.format == py::format_descriptor<float>::format())
                {
                    v = glm::make_vec3(static_cast<float*>(info.ptr));
                }
                else if(info.format == py::format_descriptor<double>::format())
                {
                    v = glm::make_vec3(static_cast<double*>(info.ptr));
                }

                return v;
            }))
        .def_buffer(
            [](glm::vec3& v) -> py::buffer_info
            {
                return py::buffer_info(glm::value_ptr(v),
                                       sizeof(float),
                                       py::format_descriptor<float>::format(),
                                       1,
                                       {3},
                                       {sizeof(float)});
            });

    py::class_<glm::mat3>(m, "Matrix3", py::buffer_protocol())
        .def(py::init(
            [](py::buffer b)
            {
                py::buffer_info info = b.request();

                glm::mat3 M;

                if(info.format == py::format_descriptor<float>::format())
                {
                    M = glm::make_mat3(static_cast<float*>(info.ptr));
                }
                else if(info.format == py::format_descriptor<double>::format())
                {
                    M = glm::make_mat3(static_cast<double*>(info.ptr));
                }

                return M;
            }))
        .def_buffer(
            [](glm::mat3& M) -> py::buffer_info
            {
                return py::buffer_info(glm::value_ptr(M),
                                       sizeof(float),
                                       py::format_descriptor<float>::format(),
                                       2,
                                       {3, 3},
                                       {sizeof(float) * 3, sizeof(float)});
            });

    py::class_<glm::mat4>(m, "Matrix4", py::buffer_protocol())
        .def(py::init(
            [](py::buffer b)
            {
                py::buffer_info info = b.request();

                glm::mat4 M;

                if(info.format == py::format_descriptor<float>::format())
                {
                    M = glm::make_mat4(static_cast<float*>(info.ptr));
                }
                else if(info.format == py::format_descriptor<double>::format())
                {
                    M = glm::make_mat4(static_cast<double*>(info.ptr));
                }

                return glm::transpose(M);
            }))
        .def_buffer(
            [](glm::mat4& M) -> py::buffer_info
            {
                return py::buffer_info(glm::value_ptr(M),
                                       sizeof(float),
                                       py::format_descriptor<float>::format(),
                                       2,
                                       {4, 4},
                                       {sizeof(float), sizeof(float) * 4});
            });

    py::class_<atcg::PerspectiveCamera, std::shared_ptr<atcg::PerspectiveCamera>>(m, "PerspectiveCamera")
        .def("getPosition", &atcg::PerspectiveCamera::getPosition)
        .def("setPosition", &atcg::PerspectiveCamera::setPosition)
        .def("getView", &atcg::PerspectiveCamera::getView)
        .def("setView", &atcg::PerspectiveCamera::setView);

    py::class_<atcg::CameraController>(m, "CameraController")
        .def(py::init<float>())
        .def("onUpdate", &atcg::CameraController::onUpdate)
        .def("onEvent", &atcg::CameraController::onEvent)
        .def("getCamera", &atcg::CameraController::getCamera);

    py::class_<atcg::Shader, std::shared_ptr<atcg::Shader>>(m, "Shader");

    py::class_<atcg::ShaderManager>(m, "ShaderManager").def_static("getShader", &atcg::ShaderManager::getShader);

    py::class_<atcg::Mesh, std::shared_ptr<atcg::Mesh>>(m, "Mesh").def("uploadData", &atcg::Mesh::uploadData);

    m.def("readMesh", &atcg::IO::read_mesh);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}