#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>
#include <Core/EntryPoint.h>
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

int python_main(atcg::Layer* layer)
{
    atcg::Application* app = atcg::createApplication();
    app->pushLayer(layer);
    return atcg::atcg_main(app);
}

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_DECLARE_HOLDER_TYPE(T, atcg::ref_ptr<T>);

PYBIND11_MODULE(pyatcg, m)
{
    m.doc() = R"pbdoc(
        Pybind11 atcg plugin
        -----------------------
        .. currentmodule:: pyatcg
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    // ---------------- CORE ---------------------
    auto m_layer  = py::class_<atcg::Layer, PythonLayer, std::unique_ptr<atcg::Layer, py::nodelete>>(m, "Layer");
    auto m_event  = py::class_<atcg::Event>(m, "Event");
    auto m_camera = py::class_<atcg::PerspectiveCamera, atcg::ref_ptr<atcg::PerspectiveCamera>>(m, "PerspectiveCamera");
    auto m_controller =
        py::class_<atcg::FirstPersonController, atcg::ref_ptr<atcg::FirstPersonController>>(m, "FirstPersonController");
    auto m_entity  = py::class_<atcg::Entity>(m, "Entity");
    auto m_scene   = py::class_<atcg::Scene, atcg::ref_ptr<atcg::Scene>>(m, "Scene");
    auto m_vec2    = py::class_<glm::vec2>(m, "vec2", py::buffer_protocol());
    auto m_ivec2   = py::class_<glm::ivec2>(m, "ivec2", py::buffer_protocol());
    auto m_vec3    = py::class_<glm::vec3>(m, "vec3", py::buffer_protocol());
    auto m_ivec3   = py::class_<glm::ivec3>(m, "ivec3", py::buffer_protocol());
    auto m_u32vec3 = py::class_<glm::u32vec3>(m, "u32vec3", py::buffer_protocol());
    auto m_vec4    = py::class_<glm::vec4>(m, "vec4", py::buffer_protocol());
    auto m_ivec4   = py::class_<glm::ivec4>(m, "ivec4", py::buffer_protocol());
    auto m_mat3    = py::class_<glm::mat3>(m, "mat3", py::buffer_protocol());
    auto m_mat4    = py::class_<glm::mat4>(m, "mat4", py::buffer_protocol());

    m.def("show", &python_main, "layer"_a, "Start the application.");
    m.def("print_statistics", &atcg::print_statistics);
    m_layer.def(py::init<>())
        .def(py::init<std::string>(), "name"_a)
        .def("onAttach", &atcg::Layer::onAttach)
        .def("onUpdate", &atcg::Layer::onUpdate, "delta_time"_a)
        .def("onImGuiRender", &atcg::Layer::onImGuiRender)
        .def("onEvent", &atcg::Layer::onEvent, "event"_a);

    m_event.def("getName", &atcg::Event::getName).def_readwrite("handled", &atcg::Event::handled);

    py::class_<atcg::WindowCloseEvent, atcg::Event>(m, "WindowCloseEvent");
    py::class_<atcg::WindowResizeEvent, atcg::Event>(m, "WindowResizeEvent")
        .def(py::init<unsigned int, unsigned int>(), "width"_a, "height"_a)
        .def("getWidth", &atcg::WindowResizeEvent::getWidth)
        .def("getHeight", &atcg::WindowResizeEvent::getHeight);
    py::class_<atcg::MouseButtonEvent, atcg::Event>(m, "MouseButtonEvent")
        .def("getMouseButton", &atcg::MouseButtonEvent::getMouseButton)
        .def("getX", &atcg::MouseButtonEvent::getX)
        .def("getY", &atcg::MouseButtonEvent::getY);
    py::class_<atcg::MouseButtonPressedEvent, atcg::MouseButtonEvent>(m, "MouseButtonPressedEvent")
        .def(py::init<int32_t, float, float>(), "button"_a, "x"_a, "y"_a);
    py::class_<atcg::MouseButtonReleasedEvent, atcg::MouseButtonEvent>(m, "MouseButtonReleasedEvent")
        .def(py::init<int32_t, float, float>(), "button"_a, "x"_a, "y"_a);
    py::class_<atcg::MouseMovedEvent, atcg::Event>(m, "MouseMovedEvent")
        .def(py::init<float, float>(), "x"_a, "y"_a)
        .def("getX", &atcg::MouseMovedEvent::getX)
        .def("getY", &atcg::MouseMovedEvent::getY);
    py::class_<atcg::MouseScrolledEvent, atcg::Event>(m, "MouseScrolledEvent")
        .def(py::init<float, float>(), "x"_a, "y"_a)
        .def("getXOffset", &atcg::MouseScrolledEvent::getXOffset)
        .def("getYOffset", &atcg::MouseScrolledEvent::getYOffset);
    py::class_<atcg::KeyEvent, atcg::Event>(m, "KeyEvent").def("getKeyCode", &atcg::KeyEvent::getKeyCode);
    py::class_<atcg::KeyPressedEvent, atcg::KeyEvent>(m, "KeyPressedEvent")
        .def(py::init<int32_t, bool>(), "key"_a, "key_pressed"_a)
        .def("isRepeat", &atcg::KeyPressedEvent::IsRepeat)
        .def("getCode", &atcg::KeyPressedEvent::getCode);
    py::class_<atcg::KeyReleasedEvent, atcg::KeyEvent>(m, "KeyReleasedEvent").def(py::init<int32_t>());
    py::class_<atcg::KeyTypedEvent, atcg::KeyEvent>(m, "KeyTypedEvent").def(py::init<int32_t>());
    py::class_<atcg::ViewportResizeEvent, atcg::Event>(m, "ViewportResizeEvent")
        .def(py::init<unsigned int, unsigned int>(), "width"_a, "height"_a)
        .def("getWidth", &atcg::ViewportResizeEvent::getWidth)
        .def("getHeight", &atcg::ViewportResizeEvent::getHeight);

    py::class_<atcg::Application, atcg::ref_ptr<atcg::Application>>(m, "Application");

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

    m.def("getViewportSize", []() { return atcg::Application::get()->getViewportSize(); });
    m.def("getViewportPosition", []() { return atcg::Application::get()->getViewportPosition(); });

    m.def(
        "enableDockSpace",
        [](bool enable) { atcg::Application::get()->enableDockSpace(enable); },
        "enable"_a);

    // ---------------- MATH -------------------------
    m_vec2.def(py::init<float, float>(), "x"_a, "y"_a)
        .def(py::init<float>(), "value"_a)
        .def(py::init(
                 [](py::array_t<float> b)
                 {
                     py::buffer_info info = b.request();

                     // Copy for now, is there a better method?
                     glm::vec2 v = glm::make_vec2(static_cast<float*>(info.ptr));

                     return v;
                 }),
             "array"_a)
        .def_buffer(
            [](glm::vec2& v) -> py::buffer_info
            {
                return py::buffer_info(glm::value_ptr(v),
                                       sizeof(float),
                                       py::format_descriptor<float>::format(),
                                       1,
                                       {2},
                                       {sizeof(float)});
            })
        .def_readwrite("x", &glm::vec2::x)
        .def_readwrite("y", &glm::vec2::y);

    m_ivec2.def(py::init<int, int>(), "x"_a, "y"_a)
        .def(py::init<int>(), "value"_a)
        .def(py::init(
                 [](py::array_t<int> b)
                 {
                     py::buffer_info info = b.request();

                     // Copy for now, is there a better method?
                     glm::ivec2 v = glm::make_vec2(static_cast<int*>(info.ptr));

                     return v;
                 }),
             "array"_a)
        .def_buffer(
            [](glm::ivec2& v) -> py::buffer_info
            {
                return py::buffer_info(glm::value_ptr(v),
                                       sizeof(int),
                                       py::format_descriptor<int>::format(),
                                       1,
                                       {2},
                                       {sizeof(int)});
            })
        .def_readwrite("x", &glm::ivec2::x)
        .def_readwrite("y", &glm::ivec2::y);

    m_vec3.def(py::init<float, float, float>(), "x"_a, "y"_a, "z"_a)
        .def(py::init<float>(), "value"_a)
        .def(py::init(
                 [](py::array_t<float> b)
                 {
                     py::buffer_info info = b.request();

                     // Copy for now, is there a better method?
                     glm::vec3 v = glm::make_vec3(static_cast<float*>(info.ptr));

                     return v;
                 }),
             "array"_a)
        .def_buffer(
            [](glm::vec3& v) -> py::buffer_info
            {
                return py::buffer_info(glm::value_ptr(v),
                                       sizeof(float),
                                       py::format_descriptor<float>::format(),
                                       1,
                                       {3},
                                       {sizeof(float)});
            })
        .def_readwrite("x", &glm::vec3::x)
        .def_readwrite("y", &glm::vec3::y)
        .def_readwrite("z", &glm::vec3::z);

    m_ivec3.def(py::init<int, int, int>(), "x"_a, "y"_a, "z"_a)
        .def(py::init<int>(), "value"_a)
        .def(py::init(
                 [](py::array_t<int> b)
                 {
                     py::buffer_info info = b.request();

                     // Copy for now, is there a better method?
                     glm::ivec3 v = glm::make_vec3(static_cast<int*>(info.ptr));

                     return v;
                 }),
             "array"_a)
        .def_buffer(
            [](glm::ivec3& v) -> py::buffer_info
            {
                return py::buffer_info(glm::value_ptr(v),
                                       sizeof(int),
                                       py::format_descriptor<int>::format(),
                                       1,
                                       {3},
                                       {sizeof(int)});
            })
        .def_readwrite("x", &glm::ivec3::x)
        .def_readwrite("y", &glm::ivec3::y)
        .def_readwrite("z", &glm::ivec3::z);

    m_u32vec3.def(py::init<uint32_t, uint32_t, uint32_t>(), "x"_a, "y"_a, "z"_a)
        .def(py::init<uint32_t>(), "value"_a)
        .def(py::init(
                 [](py::array_t<uint32_t> b)
                 {
                     py::buffer_info info = b.request();

                     // Copy for now, is there a better method?
                     glm::u32vec3 v = glm::make_vec3(static_cast<uint32_t*>(info.ptr));

                     return v;
                 }),
             "array"_a)
        .def_buffer(
            [](glm::u32vec3& v) -> py::buffer_info
            {
                return py::buffer_info(glm::value_ptr(v),
                                       sizeof(int),
                                       py::format_descriptor<uint32_t>::format(),
                                       1,
                                       {3},
                                       {sizeof(int)});
            })
        .def_readwrite("x", &glm::u32vec3::x)
        .def_readwrite("y", &glm::u32vec3::y)
        .def_readwrite("z", &glm::u32vec3::z);

    m_vec4
        .def(py::init(
                 [](py::array_t<float> b)
                 {
                     py::buffer_info info = b.request();

                     // Copy for now, is there a better method?
                     glm::vec4 v = glm::make_vec4(static_cast<float*>(info.ptr));

                     return v;
                 }),
             "array"_a)
        .def(py::init<float, float, float, float>(), "x"_a, "y"_a, "z"_a, "w"_a)
        .def(py::init<float>(), "value"_a)
        .def_buffer(
            [](glm::vec4& v) -> py::buffer_info
            {
                return py::buffer_info(glm::value_ptr(v),
                                       sizeof(float),
                                       py::format_descriptor<float>::format(),
                                       1,
                                       {4},
                                       {sizeof(float)});
            })
        .def_readwrite("x", &glm::vec4::x)
        .def_readwrite("y", &glm::vec4::y)
        .def_readwrite("z", &glm::vec4::z)
        .def_readwrite("w", &glm::vec4::w);

    m_ivec4.def(py::init<int, int, int, int>(), "x"_a, "y"_a, "z"_a, "w"_a)
        .def(py::init<int>(), "value"_a)
        .def(py::init(
                 [](py::array_t<int> b)
                 {
                     py::buffer_info info = b.request();

                     // Copy for now, is there a better method?
                     glm::ivec4 v = glm::make_vec4(static_cast<int*>(info.ptr));

                     return v;
                 }),
             "array"_a)
        .def_buffer(
            [](glm::ivec4& v) -> py::buffer_info
            {
                return py::buffer_info(glm::value_ptr(v),
                                       sizeof(int),
                                       py::format_descriptor<int>::format(),
                                       1,
                                       {4},
                                       {sizeof(int)});
            })
        .def_readwrite("x", &glm::ivec4::x)
        .def_readwrite("y", &glm::ivec4::y)
        .def_readwrite("z", &glm::ivec4::z)
        .def_readwrite("w", &glm::ivec4::w);

    m_mat3
        .def(py::init(
                 [](py::array_t<float> b)
                 {
                     py::buffer_info info = b.request();

                     glm::mat3 M = glm::make_mat3(static_cast<float*>(info.ptr));

                     return M;
                 }),
             "array"_a)
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

    m_mat4
        .def(py::init(
                 [](py::array_t<float> b)
                 {
                     py::buffer_info info = b.request();

                     glm::mat4 M = glm::make_mat4(static_cast<float*>(info.ptr));

                     return M;
                 }),
             "array"_a)
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

    // ------------------- Datastructure ---------------------------------

    py::class_<atcg::Timer>(m, "Timer")
        .def(py::init<>())
        .def("ellapsedMillis", &atcg::Timer::elapsedMillis)
        .def("ellapsedSeconds", &atcg::Timer::elapsedSeconds)
        .def("reset", &atcg::Timer::reset);

    py::class_<atcg::Graph, atcg::ref_ptr<atcg::Graph>>(m, "Graph")
        .def(py::init<>())
        .def_static("createPointCloud", py::overload_cast<>(&atcg::Graph::createPointCloud))
        .def_static("createPointCloud",
                    [](py::array_t<float> vertex_data)
                    {
                        py::buffer_info info = vertex_data.request();
                        std::vector<atcg::Vertex> vertices((atcg::Vertex*)info.ptr,
                                                           (atcg::Vertex*)info.ptr +
                                                               info.size * sizeof(float) / sizeof(atcg::Vertex));
                        return atcg::Graph::createPointCloud(vertices);
                    })
        .def("updateVertices",
             [](const atcg::ref_ptr<atcg::Graph>& graph, py::array_t<float> vertex_data)
             {
                 py::buffer_info info = vertex_data.request();
                 std::vector<atcg::Vertex> vertices((atcg::Vertex*)info.ptr,
                                                    (atcg::Vertex*)info.ptr +
                                                        info.size * sizeof(float) / sizeof(atcg::Vertex));
                 graph->updateVertices(vertices);
             })
        .def("getPositions",
             [](const atcg::ref_ptr<atcg::Graph>& graph)
             {
                 atcg::Vertex* vertices = graph->getVerticesBuffer()->getHostPointer<atcg::Vertex>();
                 return py::memoryview::from_buffer((void*)vertices,
                                                    sizeof(float),
                                                    py::format_descriptor<float>::value,
                                                    {(int)graph->n_vertices(), 3},
                                                    {sizeof(atcg::Vertex), sizeof(float)});
             })
        .def("getFaces",
             [](const atcg::ref_ptr<atcg::Graph>& graph)
             {
                 uint32_t* faces = graph->getFaceIndexBuffer()->getHostPointer<uint32_t>();
                 return py::memoryview::from_buffer((void*)faces,
                                                    sizeof(uint32_t),
                                                    py::format_descriptor<uint32_t>::value,
                                                    {(int)graph->n_faces(), 3},
                                                    {sizeof(uint32_t) * 3, sizeof(uint32_t)});
             })
        .def("unmapPointers",
             [](const atcg::ref_ptr<atcg::Graph>& graph)
             {
                 graph->getVerticesBuffer()->unmapPointers();
                 graph->getFaceIndexBuffer()->unmapPointers();
                 graph->getEdgesBuffer()->unmapPointers();
             });
    m.def(
        "read_mesh",
        [](const std::string& path) { return atcg::IO::read_mesh(path); },
        "path"_a);


    m_camera
        .def(py::init<>([](float aspect_ratio) { return atcg::make_ref<atcg::PerspectiveCamera>(aspect_ratio); }),
             "aspect_ratio"_a)
        .def("getPosition", &atcg::PerspectiveCamera::getPosition)
        .def("setPosition", &atcg::PerspectiveCamera::setPosition)
        .def("getView", &atcg::PerspectiveCamera::getView)
        .def("setView", &atcg::PerspectiveCamera::setView)
        .def("getProjection", &atcg::PerspectiveCamera::getProjection)
        .def("setProjection", &atcg::PerspectiveCamera::setProjection);

    m_controller
        .def(py::init<>([](float aspect_ratio) { return atcg::make_ref<atcg::FirstPersonController>(aspect_ratio); }))
        .def("onUpdate", &atcg::FirstPersonController::onUpdate, "delta_time"_a)
        .def("onEvent", &atcg::FirstPersonController::onEvent, "event"_a)
        .def("getCamera", &atcg::FirstPersonController::getCamera);

    // ------------------- RENDERER ---------------------------------
    py::class_<atcg::Renderer>(m, "Renderer")
        .def_static("setClearColor", &atcg::Renderer::setClearColor, "color"_a)
        .def_static("clear", &atcg::Renderer::clear)
        .def_static(
            "draw",
            [](const atcg::ref_ptr<atcg::Scene>& scene, const atcg::ref_ptr<atcg::PerspectiveCamera>& camera)
            { atcg::Renderer::draw(scene, camera); },
            "scene"_a,
            "camera"_a)
        .def_static(
            "drawCADGrid",
            [](const atcg::ref_ptr<atcg::PerspectiveCamera>& camera) { atcg::Renderer::drawCADGrid(camera); },
            "camera"_a)
        .def_static(
            "drawCameras",
            [](const atcg::ref_ptr<atcg::Scene>& scene, const atcg::ref_ptr<atcg::PerspectiveCamera>& camera)
            { atcg::Renderer::drawCameras(scene, camera); },
            "scene"_a,
            "camera"_a)
        .def_static("getEntityIndex", &atcg::Renderer::getEntityIndex, "mouse_pos"_a)
        .def_static("toggleCulling", &atcg::Renderer::toggleCulling, "enabled"_a);

    py::class_<atcg::Shader, atcg::ref_ptr<atcg::Shader>>(m, "Shader")
        .def(py::init<std::string, std::string>(), "vertex_path"_a, "fragment_path"_a)
        .def(py::init<std::string, std::string, std::string>(), "vertex_path"_a, "fragment_path"_a, "geometry_path"_a)
        .def("use", &atcg::Shader::use)
        .def("setInt", &atcg::Shader::setInt, "uniform_name"_a, "value"_a)
        .def("setFloat", &atcg::Shader::setFloat, "uniform_name"_a, "value"_a)
        .def("setVec3", &atcg::Shader::setVec3, "uniform_name"_a, "value"_a)
        .def("setVec4", &atcg::Shader::setVec4, "uniform_name"_a, "value"_a)
        .def("setMat4", &atcg::Shader::setMat4, "uniform_name"_a, "value"_a)
        .def("setMVP", &atcg::Shader::setMVP, "model"_a, "view"_a, "projection"_a);

    py::class_<atcg::ShaderManager>(m, "ShaderManager")
        .def_static("getShader", &atcg::ShaderManager::getShader, "name"_a)
        .def_static("addShader", &atcg::ShaderManager::addShader, "name"_a, "shader"_a)
        .def_static("addShaderFromName", &atcg::ShaderManager::addShaderFromName, "name"_a);

    py::enum_<atcg::TextureFormat>(m, "TextureFormat")
        .value("RGBA", atcg::TextureFormat::RGBA)
        .value("RINT", atcg::TextureFormat::RINT)
        .value("RFLOAT", atcg::TextureFormat::RFLOAT)
        .value("DEPTH", atcg::TextureFormat::DEPTH);

    py::enum_<atcg::TextureWrapMode>(m, "TextureWrapMode")
        .value("REPEAT", atcg::TextureWrapMode::REPEAT)
        .value("CLAMP_TO_EDGE", atcg::TextureWrapMode::CLAMP_TO_EDGE);

    py::enum_<atcg::TextureFilterMode>(m, "TextureFilterMode")
        .value("NEAREST", atcg::TextureFilterMode::NEAREST)
        .value("LINEAR", atcg::TextureFilterMode::LINEAR);

    py::class_<atcg::TextureSampler>(m, "TextureSampler")
        .def(py::init<>())
        .def(py::init<>(
                 [](atcg::TextureFilterMode filter_mode, atcg::TextureWrapMode wrap_mode)
                 {
                     atcg::TextureSampler sampler;
                     sampler.filter_mode = filter_mode;
                     sampler.wrap_mode   = wrap_mode;
                     return sampler;
                 }),
             "filter_mode"_a = atcg::TextureFilterMode::LINEAR,
             "wrap_mode"_a   = atcg::TextureWrapMode::REPEAT)
        .def_readwrite("wrap_mode", &atcg::TextureSampler::wrap_mode)
        .def_readwrite("filter_mode", &atcg::TextureSampler::filter_mode);

    py::class_<atcg::TextureSpecification>(m, "TextureSpecification")
        .def(py::init<>())
        .def(py::init<>(
                 [](uint32_t width,
                    uint32_t height,
                    uint32_t depth,
                    atcg::TextureSampler sampler,
                    atcg::TextureFormat format)
                 {
                     atcg::TextureSpecification spec;
                     spec.width   = width;
                     spec.height  = height;
                     spec.depth   = depth;
                     spec.sampler = sampler;
                     spec.format  = format;
                     return spec;
                 }),
             "width"_a,
             "height"_a,
             "depth"_a,
             "sampler"_a,
             "format"_a)
        .def_readwrite("width", &atcg::TextureSpecification::width)
        .def_readwrite("height", &atcg::TextureSpecification::height)
        .def_readwrite("depth", &atcg::TextureSpecification::depth)
        .def_readwrite("sampler", &atcg::TextureSpecification::sampler)
        .def_readwrite("format", &atcg::TextureSpecification::format);

    py::class_<atcg::Texture2D, atcg::ref_ptr<atcg::Texture2D>>(m, "Texture2D")
        .def_static(
            "create",
            [](atcg::TextureSpecification spec) { return atcg::Texture2D::create(spec); },
            "specification"_a)
        .def("getID", &atcg::Texture2D::getID)
        .def(
            "setData",
            [](const atcg::ref_ptr<atcg::Texture2D>& texture, py::array_t<uint8_t> b)
            {
                py::buffer_info info = b.request();
                texture->setData(info.ptr);
            },
            "data"_a);

    // ------------------- Scene ---------------------------------
    py::class_<entt::entity>(m, "EntityHandle").def(py::init<uint32_t>(), "handle"_a);

    py::class_<atcg::TransformComponent>(m, "TransformComponent")
        .def(py::init<glm::vec3, glm::vec3, glm::vec3>(), "position"_a, "scale"_a, "rotation"_a)
        .def(py::init<glm::mat4>(), "model"_a)
        .def("setPosition", &atcg::TransformComponent::setPosition, "position"_a)
        .def("setRotation", &atcg::TransformComponent::setRotation, "rotation"_a)
        .def("setScale", &atcg::TransformComponent::setScale, "scale"_a)
        .def("setModel", &atcg::TransformComponent::setModel, "model"_a)
        .def("getPosition", &atcg::TransformComponent::getPosition)
        .def("getRotation", &atcg::TransformComponent::getRotation)
        .def("getScale", &atcg::TransformComponent::getScale)
        .def("getModel", &atcg::TransformComponent::getModel);

    py::class_<atcg::GeometryComponent>(m, "GeometryComponent")
        .def(py::init<>())
        .def(py::init<const atcg::ref_ptr<atcg::Graph>&>(), "graph"_a)
        .def_readwrite("graph", &atcg::GeometryComponent::graph);

    py::class_<atcg::MeshRenderComponent>(m, "MeshRenderComponent")
        .def(py::init<>())
        .def(py::init<const atcg::ref_ptr<atcg::Shader>&>(), "shader"_a)
        .def_readwrite("visible", &atcg::MeshRenderComponent::visible)
        .def_readwrite("shader", &atcg::MeshRenderComponent::shader);

    py::class_<atcg::PointRenderComponent>(m, "PointRenderComponent")
        .def(py::init<const atcg::ref_ptr<atcg::Shader>&, glm::vec3, float>(), "shader"_a, "color"_a, "point_size"_a)
        .def_readwrite("visible", &atcg::PointRenderComponent::visible)
        .def_readwrite("color", &atcg::PointRenderComponent::color)
        .def_readwrite("shader", &atcg::PointRenderComponent::shader);

    py::class_<atcg::PointSphereRenderComponent>(m, "PointSphereRenderComponent")
        .def(py::init<const atcg::ref_ptr<atcg::Shader>&, float>(), "shader"_a, "point_size"_a)
        .def_readwrite("visible", &atcg::PointSphereRenderComponent::visible)
        .def_readwrite("shader", &atcg::PointSphereRenderComponent::shader);

    py::class_<atcg::EdgeRenderComponent>(m, "EdgeRenderComponent")
        .def(py::init<glm::vec3>(), "color"_a)
        .def_readwrite("visible", &atcg::EdgeRenderComponent::visible)
        .def_readwrite("color", &atcg::EdgeRenderComponent::color);

    py::class_<atcg::EdgeCylinderRenderComponent>(m, "EdgeCylinderRenderComponent")
        .def(py::init<float>(), "radius"_a)
        .def_readwrite("visible", &atcg::EdgeCylinderRenderComponent::visible);

    py::class_<atcg::NameComponent>(m, "NameComponent")
        .def(py::init<>())
        .def(py::init<std::string>(), "name"_a)
        .def_readwrite("name", &atcg::NameComponent::name);

    m_entity.def(py::init<>())
        .def(py::init<entt::entity, atcg::Scene*>(), "handle"_a, "scene"_a)
        .def(py::init<>([](entt::entity e, const atcg::ref_ptr<atcg::Scene>& scene)
                        { return atcg::Entity(e, scene.get()); }),
             "handle"_a,
             "scene"_a)
        .def(
            "addTransformComponent",
            [](atcg::Entity& entity, const glm::vec3& position, const glm::vec3& scale, const glm::vec3& rotation)
            { return entity.addComponent<atcg::TransformComponent>(position, scale, rotation); },
            "positiona"_a,
            "scale"_a,
            "rotation"_a)
        .def(
            "addTransformComponent",
            [](atcg::Entity& entity, const atcg::TransformComponent& transform)
            { return entity.addComponent<atcg::TransformComponent>(transform); },
            "transform"_a)
        .def(
            "addGeometryComponent",
            [](atcg::Entity& entity, const atcg::ref_ptr<atcg::Graph>& graph)
            { return entity.addComponent<atcg::GeometryComponent>(graph); },
            "graph"_a)
        .def(
            "addGeometryComponent",
            [](atcg::Entity& entity, const atcg::GeometryComponent& geometry)
            { return entity.addComponent<atcg::GeometryComponent>(geometry); },
            "geometry"_a)
        .def(
            "addMeshRenderComponent",
            [](atcg::Entity& entity, const atcg::ref_ptr<atcg::Shader>& shader)
            { return entity.addComponent<atcg::MeshRenderComponent>(shader); },
            "shader"_a)
        .def(
            "addMeshRenderComponent",
            [](atcg::Entity& entity, const atcg::MeshRenderComponent& component)
            { return entity.addComponent<atcg::MeshRenderComponent>(component); },
            "component"_a)
        .def(
            "addPointRenderComponent",
            [](atcg::Entity& entity,
               const atcg::ref_ptr<atcg::Shader>& shader,
               const glm::vec3& color,
               float point_size) { return entity.addComponent<atcg::PointRenderComponent>(shader, color, point_size); },
            "shader"_a,
            "color"_a,
            "point_size"_a)
        .def(
            "addPointRenderComponent",
            [](atcg::Entity& entity, const atcg::PointRenderComponent& component)
            { return entity.addComponent<atcg::PointRenderComponent>(component); },
            "component_a")
        .def(
            "addPointSphereRenderComponent",
            [](atcg::Entity& entity, const atcg::ref_ptr<atcg::Shader>& shader, float point_size)
            { return entity.addComponent<atcg::PointSphereRenderComponent>(shader, point_size); },
            "shader"_a,
            "point_size"_a)
        .def(
            "addPointSphereRenderComponent",
            [](atcg::Entity& entity, const atcg::PointSphereRenderComponent& component)
            { return entity.addComponent<atcg::PointSphereRenderComponent>(component); },
            "component"_a)
        .def(
            "addEdgeRenderComponent",
            [](atcg::Entity& entity, const glm::vec3& color)
            { return entity.addComponent<atcg::EdgeRenderComponent>(color); },
            "color"_a)
        .def(
            "addEdgeRenderComponent",
            [](atcg::Entity& entity, const atcg::EdgeRenderComponent& component)
            { return entity.addComponent<atcg::EdgeRenderComponent>(component); },
            "component"_a)
        .def(
            "addEdgeCylinderRenderComponent",
            [](atcg::Entity& entity, float radius)
            { return entity.addComponent<atcg::EdgeCylinderRenderComponent>(radius); },
            "radius"_a)
        .def(
            "addEdgeCylinderRenderComponent",
            [](atcg::Entity& entity, const atcg::EdgeCylinderRenderComponent& component)
            { return entity.addComponent<atcg::EdgeCylinderRenderComponent>(component); },
            "component"_a)
        .def("addNameComponent",
             [](atcg::Entity& entity, const std::string& name)
             { return entity.addComponent<atcg::NameComponent>(name); })
        .def("addNameComponent",
             [](atcg::Entity& entity, const atcg::NameComponent& component)
             { return entity.addComponent<atcg::NameComponent>(component); })
        .def("hasTransformComponent", &atcg::Entity::hasComponent<atcg::TransformComponent>)
        .def("hasGeometryComponent", &atcg::Entity::hasComponent<atcg::GeometryComponent>)
        .def("hasMeshRenderComponent", &atcg::Entity::hasComponent<atcg::MeshRenderComponent>)
        .def("hasPointRenderComponent", &atcg::Entity::hasComponent<atcg::PointRenderComponent>)
        .def("hasPointSphereRenderComponent", &atcg::Entity::hasComponent<atcg::PointSphereRenderComponent>)
        .def("hasEdgeRenderComponent", &atcg::Entity::hasComponent<atcg::EdgeRenderComponent>)
        .def("hasEdgeCylinderRenderComponent", &atcg::Entity::hasComponent<atcg::EdgeCylinderRenderComponent>)
        .def("hasNameComponent", &atcg::Entity::hasComponent<atcg::NameComponent>)
        .def("getTransformComponent", &atcg::Entity::getComponent<atcg::TransformComponent>)
        .def("getGeometryComponent", &atcg::Entity::getComponent<atcg::GeometryComponent>)
        .def("getMeshRenderComponent", &atcg::Entity::getComponent<atcg::MeshRenderComponent>)
        .def("getPointRenderComponent", &atcg::Entity::getComponent<atcg::PointRenderComponent>)
        .def("getPointSphereRenderComponent", &atcg::Entity::getComponent<atcg::PointSphereRenderComponent>)
        .def("getEdgeRenderComponent", &atcg::Entity::getComponent<atcg::EdgeRenderComponent>)
        .def("getEdgeCylinderRenderComponent", &atcg::Entity::getComponent<atcg::EdgeCylinderRenderComponent>)
        .def("getNameComponent", &atcg::Entity::getComponent<atcg::NameComponent>);

    m_scene.def(py::init<>([]() { return atcg::make_ref<atcg::Scene>(); }))
        .def("createEntity", &atcg::Scene::createEntity, "name"_a = "Entity")
        .def("getEntityByName", &atcg::Scene::getEntitiesByName, "name"_a)
        .def("getEntities",
             [](const atcg::ref_ptr<atcg::Scene>& scene)
             {
                 std::vector<atcg::Entity> entities;
                 auto view = scene->getAllEntitiesWith<atcg::IDComponent>();
                 for(auto e: view) { entities.push_back(atcg::Entity(e, scene.get())); }
                 return entities;
             });

    py::class_<atcg::SceneHierarchyPanel>(m, "SceneHierarchyPanel")
        .def(py::init<>())
        .def(py::init<const atcg::ref_ptr<atcg::Scene>&>(), "scene"_a)
        .def("renderPanel", &atcg::SceneHierarchyPanel::renderPanel)
        .def("selectEntity", &atcg::SceneHierarchyPanel::selectEntity, "entity"_a)
        .def("getSelectedEntity", &atcg::SceneHierarchyPanel::getSelectedEntity);

    // m.def("width",
    //       []()
    //       {
    //           const auto& window = atcg::Application::get()->getWindow();
    //           return (float)window->getWidth();
    //       });

    // m.def("height",
    //       []()
    //       {
    //           const auto& window = atcg::Application::get()->getWindow();
    //           return (float)window->getHeight();
    //       });

    // m.def("setSize",
    //       [](uint32_t width, uint32_t height)
    //       {
    //           const auto& window = atcg::Application::get()->getWindow();
    //           window->resize(width, height);
    //       });


    // py::class_<atcg::Input>(m, "Input")
    //     .def_static("isKeyPressed", &atcg::Input::isKeyPressed)
    //     .def_static("isMouseButtonPressed", &atcg::Input::isMouseButtonPressed)
    //     .def("getMousePosition",
    //          []()
    //          {
    //              glm::vec2 mouse_position = atcg::Input::getMousePosition();
    //              return py::array(2, reinterpret_cast<float*>(&mouse_position));
    //          });


    // py::class_<glm::vec3>(m, "Vector3", py::buffer_protocol())
    //     .def(py::init<float, float, float>())
    //     .def(py::init(
    //         [](py::buffer b)
    //         {
    //             py::buffer_info info = b.request();

    //             // Copy for now, is there a better method?
    //             glm::vec3 v;
    //             if(info.format == py::format_descriptor<float>::format())
    //             {
    //                 v = glm::make_vec3(static_cast<float*>(info.ptr));
    //             }
    //             else if(info.format == py::format_descriptor<double>::format())
    //             {
    //                 v = glm::make_vec3(static_cast<double*>(info.ptr));
    //             }
    //             else if(info.format == py::format_descriptor<int>::format())
    //             {
    //                 v = glm::make_vec3(static_cast<int*>(info.ptr));
    //             }

    //             return v;
    //         }))
    //     .def_buffer(
    //         [](glm::vec3& v) -> py::buffer_info
    //         {
    //             return py::buffer_info(glm::value_ptr(v),
    //                                    sizeof(float),
    //                                    py::format_descriptor<float>::format(),
    //                                    1,
    //                                    {3},
    //                                    {sizeof(float)});
    //         });

    // py::class_<glm::vec4>(m, "Vector4", py::buffer_protocol())
    //     .def(py::init(
    //         [](py::buffer b)
    //         {
    //             py::buffer_info info = b.request();

    //             // Copy for now, is there a better method?
    //             glm::vec4 v;
    //             if(info.format == py::format_descriptor<float>::format())
    //             {
    //                 v = glm::make_vec4(static_cast<float*>(info.ptr));
    //             }
    //             else if(info.format == py::format_descriptor<double>::format())
    //             {
    //                 v = glm::make_vec4(static_cast<double*>(info.ptr));
    //             }
    //             else if(info.format == py::format_descriptor<int>::format())
    //             {
    //                 v = glm::make_vec4(static_cast<int*>(info.ptr));
    //             }

    //             return v;
    //         }))
    //     .def_buffer(
    //         [](glm::vec4& v) -> py::buffer_info
    //         {
    //             return py::buffer_info(glm::value_ptr(v),
    //                                    sizeof(float),
    //                                    py::format_descriptor<float>::format(),
    //                                    1,
    //                                    {4},
    //                                    {sizeof(float)});
    //         });

    // py::class_<glm::mat3>(m, "Matrix3", py::buffer_protocol())
    //     .def(py::init(
    //         [](py::buffer b)
    //         {
    //             py::buffer_info info = b.request();

    //             glm::mat3 M;

    //             if(info.format == py::format_descriptor<float>::format())
    //             {
    //                 M = glm::make_mat3(static_cast<float*>(info.ptr));
    //             }
    //             else if(info.format == py::format_descriptor<double>::format())
    //             {
    //                 M = glm::make_mat3(static_cast<double*>(info.ptr));
    //             }
    //             else if(info.format == py::format_descriptor<int>::format())
    //             {
    //                 M = glm::make_mat3(static_cast<int*>(info.ptr));
    //             }

    //             return M;
    //         }))
    //     .def_buffer(
    //         [](glm::mat3& M) -> py::buffer_info
    //         {
    //             return py::buffer_info(glm::value_ptr(M),
    //                                    sizeof(float),
    //                                    py::format_descriptor<float>::format(),
    //                                    2,
    //                                    {3, 3},
    //                                    {sizeof(float), sizeof(float) * 3});
    //         });

    // py::class_<glm::mat4>(m, "Matrix4", py::buffer_protocol())
    //     .def(py::init(
    //         [](py::buffer b)
    //         {
    //             py::buffer_info info = b.request();

    //             glm::mat4 M;

    //             if(info.format == py::format_descriptor<float>::format())
    //             {
    //                 M = glm::make_mat4(static_cast<float*>(info.ptr));
    //             }
    //             else if(info.format == py::format_descriptor<double>::format())
    //             {
    //                 M = glm::make_mat4(static_cast<double*>(info.ptr));
    //             }
    //             else if(info.format == py::format_descriptor<int>::format())
    //             {
    //                 M = glm::make_mat4(static_cast<int*>(info.ptr));
    //             }

    //             return glm::transpose(M);
    //         }))
    //     .def_buffer(
    //         [](glm::mat4& M) -> py::buffer_info
    //         {
    //             return py::buffer_info(glm::value_ptr(M),
    //                                    sizeof(float),
    //                                    py::format_descriptor<float>::format(),
    //                                    2,
    //                                    {4, 4},
    //                                    {sizeof(float), sizeof(float) * 4});
    //         });


    // py::class_<atcg::PerspectiveCamera, atcg::ref_ptr<atcg::PerspectiveCamera>>(m, "PerspectiveCamera")
    //     .def(py::init<float>())
    //     .def("getPosition", &atcg::PerspectiveCamera::getPosition)
    //     .def("setPosition", &atcg::PerspectiveCamera::setPosition)
    //     .def("getView", &atcg::PerspectiveCamera::getView)
    //     .def("setView", &atcg::PerspectiveCamera::setView)
    //     .def("getProjection", &atcg::PerspectiveCamera::getProjection)
    //     .def("setProjection", &atcg::PerspectiveCamera::setProjection);

    // py::class_<atcg::CameraController>(m, "CameraController")
    //     .def(py::init<float>())
    //     .def("onUpdate", &atcg::CameraController::onUpdate)
    //     .def("onEvent", &atcg::CameraController::onEvent)
    //     .def("getCamera", &atcg::CameraController::getCamera);

    // py::class_<atcg::Shader, atcg::ref_ptr<atcg::Shader>>(m, "Shader")
    //     .def(py::init<std::string, std::string>())
    //     .def(py::init<std::string, std::string, std::string>())
    //     .def("use", &atcg::Shader::use)
    //     .def("setInt", &atcg::Shader::setInt)
    //     .def("setFloat", &atcg::Shader::setFloat)
    //     .def("setVec3", &atcg::Shader::setVec3)
    //     .def("setVec4", &atcg::Shader::setVec4)
    //     .def("setMat4", &atcg::Shader::setMat4)
    //     .def("setMVP", &atcg::Shader::setMVP);

    // py::class_<atcg::ShaderManager>(m, "ShaderManager")
    //     .def_static("getShader", &atcg::ShaderManager::getShader)
    //     .def_static("addShader", &atcg::ShaderManager::addShader)
    //     .def_static("addShaderFromName", &atcg::ShaderManager::addShaderFromName);

    // py::class_<atcg::Mesh, atcg::ref_ptr<atcg::Mesh>>(m, "Mesh")
    //     .def("uploadData", &atcg::Mesh::uploadData)
    //     .def("setPosition", &atcg::Mesh::setPosition)
    //     .def("setScale", &atcg::Mesh::setScale)
    //     .def("setColor", &atcg::Mesh::setColor)
    //     .def("setColors", &atcg::Mesh::setColors)
    //     .def("requestVertexColors", &atcg::Mesh::request_vertex_colors)
    //     .def("requestVertexNormals", &atcg::Mesh::request_vertex_normals);
    // py::class_<atcg::PointCloud, atcg::ref_ptr<atcg::PointCloud>>(m, "PointCloud")
    //     .def("uploadData", &atcg::PointCloud::uploadData)
    //     .def("asMatrix", &atcg::PointCloud::asMatrix)
    //     .def("fromMatrix", &atcg::PointCloud::fromMatrix)
    //     .def("setColor", &atcg::PointCloud::setColor);

    // m.def("readMesh", &atcg::IO::read_mesh);
    // m.def("readPointCloud", &atcg::IO::read_pointcloud);

    // m.def("rayMeshIntersection", &atcg::Tracing::rayMeshIntersection);

    // py::enum_<atcg::DrawMode>(m, "DrawMode")
    //     .value("ATCG_DRAW_MODE_TRIANGLE", atcg::DrawMode::ATCG_DRAW_MODE_TRIANGLE)
    //     .value("ATCG_DRAW_MODE_POINTS", atcg::DrawMode::ATCG_DRAW_MODE_POINTS)
    //     .value("ATCG_DRAW_MODE_POINTS_SPHERE", atcg::DrawMode::ATCG_DRAW_MODE_POINTS_SPHERE)
    //     .value("ATCG_DRAW_MODE_EDGES", atcg::DrawMode::ATCG_DRAW_MODE_EDGES);

    // py::class_<atcg::Renderer>(m, "Renderer")
    //     .def("init",
    //          [](uint32_t width, uint32_t height)
    //          {
    //              atcg::ref_ptr<atcg::Application> app = atcg::make_ref<atcg::Application>();
    //              const auto& window                   = app->getWindow();

    //              window->hide();
    //              window->resize(width, height);

    //              atcg::Renderer::useScreenBuffer();
    //              return app;
    //          })
    //     .def("setClearColor",
    //          [](const float r, const float g, const float b, const float a)
    //          { atcg::Renderer::setClearColor(glm::vec4(r, g, b, a)); })
    //     .def_static("setPointSize", &atcg::Renderer::setPointSize)
    //     .def_static("clear", &atcg::Renderer::clear)
    //     .def(
    //         "draw",
    //         [](const atcg::ref_ptr<atcg::Mesh>& mesh,
    //            const atcg::ref_ptr<atcg::PerspectiveCamera>& camera,
    //            const glm::vec3& color,
    //            const atcg::ref_ptr<atcg::Shader>& shader,
    //            atcg::DrawMode draw_mode) { atcg::Renderer::draw(mesh, camera, color, shader, draw_mode); },
    //         py::return_value_policy::automatic_reference)
    //     .def(
    //         "draw",
    //         [](const atcg::ref_ptr<atcg::PointCloud>& cloud,
    //            const atcg::ref_ptr<atcg::PerspectiveCamera>& camera,
    //            const glm::vec3& color,
    //            const atcg::ref_ptr<atcg::Shader>& shader)
    //         { atcg::Renderer::draw(cloud, camera, color, shader, atcg::DrawMode::ATCG_DRAW_MODE_POINTS); },
    //         py::return_value_policy::automatic_reference)
    //     .def("getFrame",
    //          []()
    //          {
    //              std::vector<uint8_t> buffer = atcg::Renderer::getFrame();
    //              return py::array(buffer.size(), buffer.data());
    //          })
    //     .def("getZBuffer",
    //          []()
    //          {
    //              std::vector<float> buffer = atcg::Renderer::getZBuffer();
    //              return py::array(buffer.size(), buffer.data());
    //          });

    // IMGUI BINDINGS

    auto mimgui = m.def_submodule("ImGui");

    mimgui.def("BeginMainMenuBar", &ImGui::BeginMainMenuBar);
    mimgui.def("EndMainMenuBar", &ImGui::EndMainMenuBar);
    mimgui.def("BeginMenu", &ImGui::BeginMenu, py::arg("label"), py::arg("enabled") = true);
    mimgui.def("EndMenu", &ImGui::EndMenu);
    mimgui.def(
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
    mimgui.def(
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
    mimgui.def("End", &ImGui::End);
    mimgui.def(
        "Checkbox",
        [](const char* label, bool* v)
        {
            auto ret = ImGui::Checkbox(label, v);
            return std::make_tuple(ret, v);
        },
        py::arg("label"),
        py::arg("v"),
        py::return_value_policy::automatic_reference);
    mimgui.def(
        "Button",
        [](const char* label)
        {
            auto ret = ImGui::Button(label);
            return ret;
        },
        py::arg("label"),
        py::return_value_policy::automatic_reference);
    mimgui.def(
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
    mimgui.def(
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
    mimgui.def(
        "Text",
        [](const char* fmt)
        {
            ImGui::Text(fmt);
            return;
        },
        py::arg("fmt"),
        py::return_value_policy::automatic_reference);

    mimgui.def(
        "Image",
        [](uint32_t textureID, uint32_t width, uint32_t height)
        {
            ImGui::Image(reinterpret_cast<void*>(textureID), ImVec2(width, height), ImVec2 {0, 1}, ImVec2 {1, 0});
            return;
        },
        py::arg("textureID"),
        py::arg("width"),
        py::arg("height"),
        py::return_value_policy::automatic_reference);

    mimgui.def("isUsing", &ImGuizmo::IsUsing);

    py::enum_<ImGuizmo::OPERATION>(mimgui, "GuizmoOperation")
        .value("TRANSLATE", ImGuizmo::OPERATION::TRANSLATE)
        .value("ROTATE", ImGuizmo::OPERATION::ROTATE)
        .value("SCALE", ImGuizmo::OPERATION::SCALE)
        .export_values();
    mimgui.def("drawGuizmo", atcg::drawGuizmo);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}