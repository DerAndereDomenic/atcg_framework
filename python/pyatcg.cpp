#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <ATCG.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>

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

    py::class_<atcg::Input>(m, "Input")
        .def_static("isKeyPressed", &atcg::Input::isKeyPressed)
        .def_static("isMouseButtonPressed", &atcg::Input::isMouseButtonPressed)
        .def("getMousePosition",
             []()
             {
                 glm::vec2 mouse_position = atcg::Input::getMousePosition();
                 return py::array(2, reinterpret_cast<float*>(&mouse_position));
             });

    py::class_<atcg::Application, std::shared_ptr<atcg::Application>>(m, "Application");

    py::class_<atcg::Renderer>(m, "Renderer")
        .def("init",
             [](uint32_t width, uint32_t height)
             {
                 std::shared_ptr<atcg::Application> app = std::make_shared<atcg::Application>();
                 const auto& window                     = app->getWindow();

                 window->hide();
                 window->resize(width, height);

                 atcg::Renderer::useScreenBuffer();
                 return app;
             })
        .def("setClearColor",
             [](const float r, const float g, const float b, const float a)
             { atcg::Renderer::setClearColor(glm::vec4(r, g, b, a)); })
        .def_static("setPointSize", &atcg::Renderer::setPointSize)
        .def_static("clear", &atcg::Renderer::clear)
        .def("renderMesh",
             [](const std::shared_ptr<atcg::Mesh>& mesh,
                const std::shared_ptr<atcg::Shader>& shader,
                const std::shared_ptr<atcg::PerspectiveCamera>& camera) { atcg::Renderer::draw(mesh, shader, camera); })
        .def("renderPoints",
             [](const std::shared_ptr<atcg::Mesh>& mesh,
                const glm::vec3& color,
                const std::shared_ptr<atcg::Shader>& shader,
                const std::shared_ptr<atcg::PerspectiveCamera>& camera)
             { atcg::Renderer::drawPoints(mesh, color, shader, camera); })
        .def("renderPointCloud",
             [](const std::shared_ptr<atcg::PointCloud>& cloud,
                const std::shared_ptr<atcg::Shader>& shader,
                const std::shared_ptr<atcg::PerspectiveCamera>& camera)
             { atcg::Renderer::draw(cloud, shader, camera); })
        .def("getFrame",
             []()
             {
                 std::vector<uint8_t> buffer = atcg::Renderer::getFrame();
                 return py::array(buffer.size(), buffer.data());
             })
        .def("getZBuffer",
             []()
             {
                 std::vector<float> buffer = atcg::Renderer::getZBuffer();
                 return py::array(buffer.size(), buffer.data());
             });

    py::class_<glm::vec3>(m, "Vector3", py::buffer_protocol())
        .def(py::init<float, float, float>())
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
                else if(info.format == py::format_descriptor<int>::format())
                {
                    v = glm::make_vec3(static_cast<int*>(info.ptr));
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

    py::class_<glm::vec4>(m, "Vector4", py::buffer_protocol())
        .def(py::init(
            [](py::buffer b)
            {
                py::buffer_info info = b.request();

                // Copy for now, is there a better method?
                glm::vec4 v;
                if(info.format == py::format_descriptor<float>::format())
                {
                    v = glm::make_vec4(static_cast<float*>(info.ptr));
                }
                else if(info.format == py::format_descriptor<double>::format())
                {
                    v = glm::make_vec4(static_cast<double*>(info.ptr));
                }
                else if(info.format == py::format_descriptor<int>::format())
                {
                    v = glm::make_vec4(static_cast<int*>(info.ptr));
                }

                return v;
            }))
        .def_buffer(
            [](glm::vec4& v) -> py::buffer_info
            {
                return py::buffer_info(glm::value_ptr(v),
                                       sizeof(float),
                                       py::format_descriptor<float>::format(),
                                       1,
                                       {4},
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
                else if(info.format == py::format_descriptor<int>::format())
                {
                    M = glm::make_mat3(static_cast<int*>(info.ptr));
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
                                       {sizeof(float), sizeof(float) * 3});
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
                else if(info.format == py::format_descriptor<int>::format())
                {
                    M = glm::make_mat4(static_cast<int*>(info.ptr));
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
        .def(py::init<float>())
        .def("getPosition", &atcg::PerspectiveCamera::getPosition)
        .def("setPosition", &atcg::PerspectiveCamera::setPosition)
        .def("getView", &atcg::PerspectiveCamera::getView)
        .def("setView", &atcg::PerspectiveCamera::setView)
        .def("getProjection", &atcg::PerspectiveCamera::getProjection)
        .def("setProjection", &atcg::PerspectiveCamera::setProjection);

    py::class_<atcg::CameraController>(m, "CameraController")
        .def(py::init<float>())
        .def("onUpdate", &atcg::CameraController::onUpdate)
        .def("onEvent", &atcg::CameraController::onEvent)
        .def("getCamera", &atcg::CameraController::getCamera);

    py::class_<atcg::Shader, std::shared_ptr<atcg::Shader>>(m, "Shader")
        .def(py::init<std::string, std::string>())
        .def(py::init<std::string, std::string, std::string>())
        .def("use", &atcg::Shader::use)
        .def("setInt", &atcg::Shader::setInt)
        .def("setFloat", &atcg::Shader::setFloat)
        .def("setVec3", &atcg::Shader::setVec3)
        .def("setVec4", &atcg::Shader::setVec4)
        .def("setMat4", &atcg::Shader::setMat4)
        .def("setMVP", &atcg::Shader::setMVP);

    py::class_<atcg::ShaderManager>(m, "ShaderManager")
        .def_static("getShader", &atcg::ShaderManager::getShader)
        .def_static("addShader", &atcg::ShaderManager::addShader)
        .def_static("addShaderFromName", &atcg::ShaderManager::addShaderFromName);

    py::class_<atcg::Mesh, std::shared_ptr<atcg::Mesh>>(m, "Mesh")
        .def("uploadData", &atcg::Mesh::uploadData)
        .def("setPosition", &atcg::Mesh::setPosition)
        .def("setScale", &atcg::Mesh::setScale)
        .def("setColor", &atcg::Mesh::setColor)
        .def("requestVertexColors", &atcg::Mesh::request_vertex_colors)
        .def("requestVertexNormals", &atcg::Mesh::request_vertex_normals);
    py::class_<atcg::PointCloud, std::shared_ptr<atcg::PointCloud>>(m, "PointCloud")
        .def("uploadData", &atcg::PointCloud::uploadData)
        .def("asMatrix", &atcg::PointCloud::asMatrix)
        .def("fromMatrix", &atcg::PointCloud::fromMatrix)
        .def("setColor", &atcg::PointCloud::setColor);

    m.def("readMesh", &atcg::IO::read_mesh);
    m.def("readPointCloud", &atcg::IO::read_pointcloud);

    m.def("rayMeshIntersection", &atcg::Tracing::rayMeshIntersection);

    // IMGUI BINDINGS

    m.def("BeginMainMenuBar", &ImGui::BeginMainMenuBar);
    m.def("EndMainMenuBar", &ImGui::EndMainMenuBar);
    m.def("BeginMenu", &ImGui::BeginMenu, py::arg("label"), py::arg("enabled") = true);
    m.def("EndMenu", &ImGui::EndMenu);
    m.def(
        "MenuItem",
        [](const char* label, const char* shortcut, bool* p_selected, bool enabled)
        {
            auto ret = ImGui::MenuItem(label, shortcut, p_selected, enabled);
            return std::make_tuple(ret, p_selected);
        },
        py::arg("label"),
        py::arg("shortcut"),
        py::arg("p_selected"),
        py::arg("enabled") = true,
        py::return_value_policy::automatic_reference);
    m.def(
        "Begin",
        [](const char* name, bool* p_open, ImGuiWindowFlags flags)
        {
            auto ret = ImGui::Begin(name, p_open, flags);
            return std::make_tuple(ret, p_open);
        },
        py::arg("name"),
        py::arg("p_open") = nullptr,
        py::arg("flags")  = 0,
        py::return_value_policy::automatic_reference);
    m.def("End", &ImGui::End);
    m.def(
        "Checkbox",
        [](const char* label, bool* v)
        {
            auto ret = ImGui::Checkbox(label, v);
            return std::make_tuple(ret, v);
        },
        py::arg("label"),
        py::arg("v"),
        py::return_value_policy::automatic_reference);
    m.def(
        "Button",
        [](const char* label)
        {
            auto ret = ImGui::Button(label);
            return ret;
        },
        py::arg("label"),
        py::return_value_policy::automatic_reference);
    m.def(
        "SliderInt",
        [](const char* label, int* v, int v_min, int v_max, const char* format = "%d", ImGuiSliderFlags flags = 0)
        {
            auto ret = ImGui::SliderInt(label, v, v_min, v_max, format, flags);
            return std::make_tuple(ret, v);
        },
        py::arg("label"),
        py::arg("v"),
        py::arg("v_min"),
        py::arg("v_max"),
        py::arg("format") = "%d",
        py::arg("flags")  = 0,
        py::return_value_policy::automatic_reference);
    m.def(
        "SliderFloat",
        [](const char* label,
           float* v,
           float v_min,
           float v_max,
           const char* format     = "%.3f",
           ImGuiSliderFlags flags = 0)
        {
            auto ret = ImGui::SliderFloat(label, v, v_min, v_max, format, flags);
            return std::make_tuple(ret, v);
        },
        py::arg("label"),
        py::arg("v"),
        py::arg("v_min"),
        py::arg("v_max"),
        py::arg("format") = "%.3f",
        py::arg("flags")  = 0,
        py::return_value_policy::automatic_reference);
    m.def(
        "Text",
        [](const char* fmt)
        {
            ImGui::Text(fmt);
            return;
        },
        py::arg("fmt"),
        py::return_value_policy::automatic_reference);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}