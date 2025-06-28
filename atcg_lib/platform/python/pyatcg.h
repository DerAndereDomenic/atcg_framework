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
#define ATCG_DEFINE_MODULES(m)                                                                                                  \
    py::class_<atcg::Application>(m, "Application");                                                                            \
    auto m_application = py::class_<PythonApplication, atcg::Application>(m, "PythonApplication");                              \
    auto m_layer       = py::class_<atcg::Layer, PythonLayer, std::unique_ptr<atcg::Layer, py::nodelete>>(m, "Layer");          \
    auto m_event       = py::class_<atcg::Event>(m, "Event");                                                                   \
    auto m_camera =                                                                                                             \
        py::class_<atcg::PerspectiveCamera, atcg::ref_ptr<atcg::PerspectiveCamera>>(m, "PerspectiveCamera");                    \
    auto m_extrinsics          = py::class_<atcg::CameraExtrinsics>(m, "CameraExtrinsics");                                     \
    auto m_intrinsics          = py::class_<atcg::CameraIntrinsics>(m, "CameraIntrinsics");                                     \
    auto m_controller          = py::class_<atcg::FirstPersonController, atcg::ref_ptr<atcg::FirstPersonController>>(m,         \
                                                                                                            "FirstPer" \
                                                                                                                     "sonContr" \
                                                                                                                     "oller");  \
    auto m_entity              = py::class_<atcg::Entity>(m, "Entity");                                                         \
    auto m_scene               = py::class_<atcg::Scene, atcg::ref_ptr<atcg::Scene>>(m, "Scene");                               \
    auto m_vec2                = py::class_<glm::vec2>(m, "vec2", py::buffer_protocol());                                       \
    auto m_ivec2               = py::class_<glm::ivec2>(m, "ivec2", py::buffer_protocol());                                     \
    auto m_vec3                = py::class_<glm::vec3>(m, "vec3", py::buffer_protocol());                                       \
    auto m_ivec3               = py::class_<glm::ivec3>(m, "ivec3", py::buffer_protocol());                                     \
    auto m_u32vec3             = py::class_<glm::u32vec3>(m, "u32vec3", py::buffer_protocol());                                 \
    auto m_vec4                = py::class_<glm::vec4>(m, "vec4", py::buffer_protocol());                                       \
    auto m_ivec4               = py::class_<glm::ivec4>(m, "ivec4", py::buffer_protocol());                                     \
    auto m_mat3                = py::class_<glm::mat3>(m, "mat3", py::buffer_protocol());                                       \
    auto m_mat4                = py::class_<glm::mat4>(m, "mat4", py::buffer_protocol());                                       \
    auto m_window_props        = py::class_<atcg::WindowProps>(m, "WindowProps");                                               \
    auto m_window_close_evnet  = py::class_<atcg::WindowCloseEvent, atcg::Event>(m, "WindowCloseEvent");                        \
    auto m_window_resize_event = py::class_<atcg::WindowResizeEvent, atcg::Event>(m, "WindowResizeEvent");                      \
    auto m_mouse_button_event  = py::class_<atcg::MouseButtonEvent, atcg::Event>(m, "MouseButtonEvent");                        \
    auto m_mouse_button_pressed_event =                                                                                         \
        py::class_<atcg::MouseButtonPressedEvent, atcg::MouseButtonEvent>(m, "MouseButtonPressedEvent");                        \
    auto m_mouse_button_released_event =                                                                                        \
        py::class_<atcg::MouseButtonReleasedEvent, atcg::MouseButtonEvent>(m, "MouseButtonReleasedEvent");                      \
    auto m_mouse_moved_event     = py::class_<atcg::MouseMovedEvent, atcg::Event>(m, "MouseMovedEvent");                        \
    auto m_mouse_scrolled_event  = py::class_<atcg::MouseScrolledEvent, atcg::Event>(m, "MouseScrolledEvent");                  \
    auto m_key_event             = py::class_<atcg::KeyEvent, atcg::Event>(m, "KeyEvent");                                      \
    auto m_key_pressed_event     = py::class_<atcg::KeyPressedEvent, atcg::KeyEvent>(m, "KeyPressedEvent");                     \
    auto m_key_released_event    = py::class_<atcg::KeyReleasedEvent, atcg::KeyEvent>(m, "KeyReleasedEvent");                   \
    auto m_key_typed_event       = py::class_<atcg::KeyTypedEvent, atcg::KeyEvent>(m, "KeyTypedEvent");                         \
    auto m_viewport_resize_event = py::class_<atcg::ViewportResizeEvent, atcg::Event>(m, "ViewportResizeEvent");                \
    auto m_timer                 = py::class_<atcg::Timer>(m, "Timer");                                                         \
    auto m_vertex_specification  = py::class_<atcg::VertexSpecification>(m, "VertexSpecification");                             \
    auto m_edge_specification    = py::class_<atcg::EdgeSpecification>(m, "EdgeSpecification");                                 \
    auto m_graph                 = py::class_<atcg::Graph, atcg::ref_ptr<atcg::Graph>>(m, "Graph");                             \
    auto m_serializer            = py::class_<atcg::Serialization::SceneSerializer>(m, "SceneSerializer");                      \
    auto m_renderer              = m.def_submodule("Renderer");                                                                 \
    auto m_renderer_system =                                                                                                    \
        py::class_<atcg::RendererSystem, atcg::ref_ptr<atcg::RendererSystem>>(m, "RendererSystem");                             \
    auto m_shader         = py::class_<atcg::Shader, atcg::ref_ptr<atcg::Shader>>(m, "Shader");                                 \
    auto m_shader_manager = m.def_submodule("ShaderManager");                                                                   \
    auto m_shader_manager_system =                                                                                              \
        py::class_<atcg::ShaderManagerSystem, atcg::ref_ptr<atcg::ShaderManagerSystem>>(m, "ShaderManagerSystem");              \
    auto m_texture_format        = py::enum_<atcg::TextureFormat>(m, "TextureFormat");                                          \
    auto m_texture_wrap_mode     = py::enum_<atcg::TextureWrapMode>(m, "TextureWrapMode");                                      \
    auto m_texture_filter_mode   = py::enum_<atcg::TextureFilterMode>(m, "TextureFilterMode");                                  \
    auto m_texture_sampler       = py::class_<atcg::TextureSampler>(m, "TextureSampler");                                       \
    auto m_texture_specification = py::class_<atcg::TextureSpecification>(m, "TextureSpecification");                           \
    auto m_image          = py::class_<atcg::Image, atcg::ref_ptr<atcg::Image>>(m, "Image", py::buffer_protocol());             \
    auto m_texture2d      = py::class_<atcg::Texture2D, atcg::ref_ptr<atcg::Texture2D>>(m, "Texture2D");                        \
    auto m_texture_cube   = py::class_<atcg::TextureCube, atcg::ref_ptr<atcg::TextureCube>>(m, "TextureCube");                  \
    auto m_framebuffer    = py::class_<atcg::Framebuffer, atcg::ref_ptr<atcg::Framebuffer>>(m, "Framebuffer");                  \
    auto m_entity_handle  = py::class_<entt::entity>(m, "EntityHandle");                                                        \
    auto m_material       = py::class_<atcg::Material>(m, "Material");                                                          \
    auto m_transform      = py::class_<atcg::TransformComponent>(m, "TransformComponent");                                      \
    auto m_geometry       = py::class_<atcg::GeometryComponent>(m, "GeometryComponent");                                        \
    auto m_mesh_renderer  = py::class_<atcg::MeshRenderComponent>(m, "MeshRenderComponent");                                    \
    auto m_point_renderer = py::class_<atcg::PointRenderComponent>(m, "PointRenderComponent");                                  \
    auto m_point_sphere_renderer  = py::class_<atcg::PointSphereRenderComponent>(m, "PointSphereRenderComponent");              \
    auto m_edge_renderer          = py::class_<atcg::EdgeRenderComponent>(m, "EdgeRenderComponent");                            \
    auto m_edge_cylinder_renderer = py::class_<atcg::EdgeCylinderRenderComponent>(m, "EdgeCylinderRenderComponent");            \
    auto m_instance_renderer      = py::class_<atcg::InstanceRenderComponent>(m, "InstanceRenderComponent");                    \
    auto m_vertex_buffer    = py::class_<atcg::VertexBuffer, atcg::ref_ptr<atcg::VertexBuffer>>(m, "VertexBuffer");             \
    auto m_buffer_layout    = py::class_<atcg::BufferLayout>(m, "BufferLayout");                                                \
    auto m_buffer_element   = py::class_<atcg::BufferElement>(m, "BufferElement");                                              \
    auto m_shader_data_type = py::enum_<atcg::ShaderDataType>(m, "ShaderDataType");                                             \
    auto m_name             = py::class_<atcg::NameComponent>(m, "NameComponent");                                              \
    auto m_point_light      = py::class_<atcg::PointLightComponent>(m, "PointLightComponent");                                  \
    auto m_script_component = py::class_<atcg::ScriptComponent>(m, "ScriptComponent");                                          \
    auto m_scene_hierarchy_panel =                                                                                              \
        py::class_<atcg::SceneHierarchyPanel<atcg::ComponentGUIHandler>>(m, "SceneHierarchyPanel");                             \
    auto m_hit_info          = py::class_<atcg::Tracing::HitInfo>(m, "HitInfo");                                                \
    auto m_utils             = m.def_submodule("Utils");                                                                        \
    auto m_draw_mode         = py::enum_<atcg::DrawMode>(m, "DrawMode");                                                        \
    auto m_cull_mode         = py::enum_<atcg::CullMode>(m, "CullMode");                                                        \
    auto m_network           = m.def_submodule("Network");                                                                      \
    auto m_tcp_server        = py::class_<atcg::TCPServer>(m_network, "TCPServer");                                             \
    auto m_tcp_client        = py::class_<atcg::TCPClient>(m_network, "TCPClient");                                             \
    auto m_performance_panel = py::class_<atcg::PerformancePanel>(m, "PerformancePanel");                                       \
    auto m_scriptengine =                                                                                                       \
        py::class_<atcg::PythonScriptEngine, atcg::ref_ptr<atcg::PythonScriptEngine>>(m, "ScriptEngine");                       \
    auto m_script = py::class_<atcg::PythonScript, atcg::ref_ptr<atcg::PythonScript>>(m, "Script");

inline void defineBindings(py::module_& m)
{
    namespace py = pybind11;
    using namespace pybind11::literals;
    m.doc() = R"pbdoc(
        Pybind11 atcg plugin
        -----------------------
        .. currentmodule:: pyatcg
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    // ---------------- CORE ---------------------
    ATCG_DEFINE_MODULES(m)
#ifndef ATCG_HEADLESS
    auto m_imgui            = m.def_submodule("ImGui");
    auto m_guizmo_operation = py::enum_<ImGuizmo::OPERATION>(m_imgui, "GuizmoOperation");
#endif

// On module initialization and destruction
#ifdef ATCG_PYTHON_MODULE
    py::class_<PythonContext>(m, "PythonContext").def(py::init<>());
    static PythonContext context;

    py::module atexit = py::module::import("atexit");
    atexit.attr("register")(py::cpp_function([]() { context.onExit(); }));
#endif

    // py::object python_context = py::cast(new PythonContext());
    // m.attr("_python_context") = python_context;

    m_window_props.def(py::init<>([]() { return atcg::WindowProps(); }))
        .def(py::init<const std::string&, uint32_t, uint32_t, int32_t, int32_t, bool>())
        .def_readwrite("tile", &atcg::WindowProps::title)
        .def_readwrite("width", &atcg::WindowProps::width)
        .def_readwrite("height", &atcg::WindowProps::height)
        .def_readwrite("pos_x", &atcg::WindowProps::pos_x)
        .def_readwrite("pos_y", &atcg::WindowProps::pos_y)
        .def_readwrite("vsync", &atcg::WindowProps::vsync)
        .def_readwrite("hidden", &atcg::WindowProps::hidden);

    m.def("init",
          []()
          {
              atcg::WindowProps props;
              props.hidden                           = true;
              std::unique_ptr<PythonApplication> app = std::make_unique<PythonApplication>(props);
              return app;
          });

    m.def("shader_directory", []() { return atcg::shader_directory().string(); });
    m.def("resource_directory", []() { return atcg::resource_directory().string(); });

    m_application.def(py::init<atcg::Layer*>())
        .def(py::init<atcg::WindowProps>())
        .def(py::init<atcg::Layer*, atcg::WindowProps>())
        .def("run", &atcg::Application::run);
    m_layer.def(py::init<>())
        .def(py::init<std::string>(), "name"_a)
        .def("onAttach", &atcg::Layer::onAttach)
        .def("onUpdate", &atcg::Layer::onUpdate, "delta_time"_a)
        .def("onImGuiRender", &atcg::Layer::onImGuiRender)
        .def("onEvent", &atcg::Layer::onEvent, "event"_a);

    m_event.def("getName", &atcg::Event::getName).def_readwrite("handled", &atcg::Event::handled);

    m_window_resize_event.def(py::init<unsigned int, unsigned int>(), "width"_a, "height"_a)
        .def("getWidth", &atcg::WindowResizeEvent::getWidth)
        .def("getHeight", &atcg::WindowResizeEvent::getHeight);
    m_mouse_button_event.def("getMouseButton", &atcg::MouseButtonEvent::getMouseButton)
        .def("getX", &atcg::MouseButtonEvent::getX)
        .def("getY", &atcg::MouseButtonEvent::getY);
    m_mouse_button_pressed_event.def(py::init<int32_t, float, float>(), "button"_a, "x"_a, "y"_a);
    m_mouse_button_released_event.def(py::init<int32_t, float, float>(), "button"_a, "x"_a, "y"_a);
    m_mouse_moved_event.def(py::init<float, float>(), "x"_a, "y"_a)
        .def("getX", &atcg::MouseMovedEvent::getX)
        .def("getY", &atcg::MouseMovedEvent::getY);
    m_mouse_scrolled_event.def(py::init<float, float>(), "x"_a, "y"_a)
        .def("getXOffset", &atcg::MouseScrolledEvent::getXOffset)
        .def("getYOffset", &atcg::MouseScrolledEvent::getYOffset);
    m_key_event.def("getKeyCode", &atcg::KeyEvent::getKeyCode);
    m_key_pressed_event.def(py::init<int32_t, bool>(), "key"_a, "key_pressed"_a)
        .def("isRepeat", &atcg::KeyPressedEvent::IsRepeat)
        .def("getCode", &atcg::KeyPressedEvent::getKeyCode);
    m_key_released_event.def(py::init<int32_t>());
    m_key_typed_event.def(py::init<int32_t>());
    m_viewport_resize_event.def(py::init<unsigned int, unsigned int>(), "width"_a, "height"_a)
        .def("getWidth", &atcg::ViewportResizeEvent::getWidth)
        .def("getHeight", &atcg::ViewportResizeEvent::getHeight);

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

    m.def("enableDockSpace", [](bool enable) { atcg::Application::get()->enableDockSpace(enable); }, "enable"_a);

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
        .def("numpy",
             [](glm::vec2& v) -> py::array_t<float>
             { return py::array_t<float>({2}, {sizeof(float)}, glm::value_ptr(v)); })
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
        .def("numpy",
             [](glm::ivec2& v) -> py::array_t<int> { return py::array_t<int>({2}, {sizeof(int)}, glm::value_ptr(v)); })
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
        .def("numpy",
             [](glm::vec3& v) -> py::array_t<float>
             { return py::array_t<float>({3}, {sizeof(float)}, glm::value_ptr(v)); })
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
        .def("numpy",
             [](glm::ivec3& v) -> py::array_t<float>
             { return py::array_t<int>({3}, {sizeof(int)}, glm::value_ptr(v)); })
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
        .def("numpy",
             [](glm::u32vec3& v) -> py::array_t<uint32_t>
             { return py::array_t<uint32_t>({3}, {sizeof(uint32_t)}, glm::value_ptr(v)); })
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
        .def("numpy",
             [](glm::vec4& v) -> py::array_t<float>
             { return py::array_t<float>({4}, {sizeof(float)}, glm::value_ptr(v)); })
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
        .def("numpy",
             [](glm::ivec4& v) -> py::array_t<int> { return py::array_t<int>({4}, {sizeof(int)}, glm::value_ptr(v)); })
        .def_readwrite("x", &glm::ivec4::x)
        .def_readwrite("y", &glm::ivec4::y)
        .def_readwrite("z", &glm::ivec4::z)
        .def_readwrite("w", &glm::ivec4::w);

    m_mat3
        .def(py::init(
                 [](py::array_t<float, py::array::c_style | py::array::forcecast> b)
                 {
                     py::buffer_info info = b.request();

                     glm::mat3 M;

                     const float* data = static_cast<const float*>(b.data());
                     for(int i = 0; i < 3; ++i)
                     {
                         for(int j = 0; j < 3; ++j)
                         {
                             M[i][j] = data[b.index_at(j, i)];
                         }
                     }

                     return M;
                 }),
             "array"_a)
        .def("numpy",
             [](glm::mat3& M) -> py::array_t<float>
             {
                 float data[3][3];

                 for(int i = 0; i < 3; ++i)
                 {
                     for(int j = 0; j < 3; ++j)
                     {
                         data[i][j] = M[j][i];
                     }
                 }

                 return py::array_t<float>({3, 3}, {sizeof(float) * 3, sizeof(float)}, (const float*)data);
             });

    m_mat4
        .def(py::init(
                 [](py::array_t<float, py::array::c_style | py::array::forcecast> b)
                 {
                     py::buffer_info info = b.request();

                     glm::mat4 M;

                     const float* data = static_cast<const float*>(b.data());
                     for(int i = 0; i < 4; ++i)
                     {
                         for(int j = 0; j < 4; ++j)
                         {
                             M[i][j] = data[b.index_at(j, i)];
                         }
                     }

                     return M;
                 }),
             "array"_a)
        .def("numpy",
             [](glm::mat4& M) -> py::array_t<float>
             {
                 float data[4][4];

                 for(int i = 0; i < 4; ++i)
                 {
                     for(int j = 0; j < 4; ++j)
                     {
                         data[i][j] = M[j][i];
                     }
                 }

                 return py::array_t<float>({4, 4}, {sizeof(float) * 4, sizeof(float)}, (const float*)data);
             });

    // ------------------- Datastructure ---------------------------------

    m_timer.def(py::init<>())
        .def("ellapsedMillis", &atcg::Timer::elapsedMillis)
        .def("ellapsedSeconds", &atcg::Timer::elapsedSeconds)
        .def("reset", &atcg::Timer::reset);

    m_vertex_specification.def_readonly_static("POSITION_BEGIN", &atcg::VertexSpecification::POSITION_BEGIN)
        .def_readonly_static("POSITION_END", &atcg::VertexSpecification::POSITION_END)
        .def_readonly_static("COLOR_BEGIN", &atcg::VertexSpecification::COLOR_BEGIN)
        .def_readonly_static("COLOR_END", &atcg::VertexSpecification::COLOR_END)
        .def_readonly_static("NORMAL_BEGIN", &atcg::VertexSpecification::NORMAL_BEGIN)
        .def_readonly_static("NORMAL_END", &atcg::VertexSpecification::NORMAL_END)
        .def_readonly_static("TANGENT_BEGIN", &atcg::VertexSpecification::TANGENT_BEGIN)
        .def_readonly_static("TANGENT_END", &atcg::VertexSpecification::TANGENT_END)
        .def_readonly_static("UV_BEGIN", &atcg::VertexSpecification::UV_BEGIN)
        .def_readonly_static("UV_END", &atcg::VertexSpecification::UV_END)
        .def_readonly_static("VERTEX_SIZE", &atcg::VertexSpecification::VERTEX_SIZE);

    m_edge_specification.def_readonly_static("INDICES_BEGIN", &atcg::EdgeSpecification::INDICES_BEGIN)
        .def_readonly_static("INDICES_END", &atcg::EdgeSpecification::INDICES_END)
        .def_readonly_static("COLOR_BEGIN", &atcg::EdgeSpecification::COLOR_BEGIN)
        .def_readonly_static("COLOR_END", &atcg::EdgeSpecification::COLOR_END)
        .def_readonly_static("RADIUS_BEGIN", &atcg::EdgeSpecification::RADIUS_BEGIN)
        .def_readonly_static("RADIUS_END", &atcg::EdgeSpecification::RADIUS_END)
        .def_readonly_static("EDGE_SIZE", &atcg::EdgeSpecification::EDGE_SIZE);

    m_graph.def(py::init<>())
        .def_static("createPointCloud", py::overload_cast<>(&atcg::Graph::createPointCloud))
        .def_static("createPointCloud", py::overload_cast<const torch::Tensor&>(&atcg::Graph::createPointCloud))
        .def_static("createTriangleMesh", py::overload_cast<>(&atcg::Graph::createTriangleMesh))
        .def_static("createTriangleMesh",
                    py::overload_cast<const torch::Tensor&, const torch::Tensor&>(&atcg::Graph::createTriangleMesh))
        .def_static("createGraph", py::overload_cast<>(&atcg::Graph::createGraph))
        .def_static("createGraph",
                    py::overload_cast<const torch::Tensor&, const torch::Tensor&>(&atcg::Graph::createGraph))
        .def("updateVertices", py::overload_cast<const torch::Tensor&>(&atcg::Graph::updateVertices))
        .def("updateFaces", py::overload_cast<const torch::Tensor&>(&atcg::Graph::updateFaces))
        .def("updateEdges", py::overload_cast<const torch::Tensor&>(&atcg::Graph::updateEdges))
        .def("getPositions", &atcg::Graph::getPositions)
        .def("getHostPositions", &atcg::Graph::getHostPositions)
        .def("getDevicePositions", &atcg::Graph::getDevicePositions)
        .def("getColors", &atcg::Graph::getColors)
        .def("getHostColors", &atcg::Graph::getHostColors)
        .def("getDeviceColors", &atcg::Graph::getDeviceColors)
        .def("getNormals", &atcg::Graph::getNormals)
        .def("getHostNormals", &atcg::Graph::getHostNormals)
        .def("getDeviceNormals", &atcg::Graph::getDeviceNormals)
        .def("getTangents", &atcg::Graph::getTangents)
        .def("getHostTangents", &atcg::Graph::getHostTangents)
        .def("getDeviceTangents", &atcg::Graph::getDeviceTangents)
        .def("getUVs", &atcg::Graph::getUVs)
        .def("getHostUVs", &atcg::Graph::getHostUVs)
        .def("getDeviceUVs", &atcg::Graph::getDeviceUVs)
        .def("getEdges", &atcg::Graph::getEdges)
        .def("getHostEdges", &atcg::Graph::getHostEdges)
        .def("getDeviceEdges", &atcg::Graph::getDeviceEdges)
        .def("n_vertices", &atcg::Graph::n_vertices)
        .def("n_faces", &atcg::Graph::n_faces)
        .def("n_edges", &atcg::Graph::n_edges)
        .def("unmapVertexPointer", &atcg::Graph::unmapVertexPointer)
        .def("unmapHostVertexPointer", &atcg::Graph::unmapHostVertexPointer)
        .def("unmapDeviceVertexPointer", &atcg::Graph::unmapDeviceVertexPointer)
        .def("unmapEdgePointer", &atcg::Graph::unmapEdgePointer)
        .def("unmapHostEdgePointer", &atcg::Graph::unmapHostEdgePointer)
        .def("unmapDeviceEdgePointer", &atcg::Graph::unmapDeviceEdgePointer)
        .def("unmapFacePointer", &atcg::Graph::unmapEdgePointer)
        .def("unmapHostFacePointer", &atcg::Graph::unmapHostFacePointer)
        .def("unmapDeviceFacePointer", &atcg::Graph::unmapDeviceFacePointer)
        .def("unmapAllHostPointers", &atcg::Graph::unmapAllHostPointers)
        .def("unmapAllDevicePointers", &atcg::Graph::unmapAllDevicePointers)
        .def("unmapAllPointers", &atcg::Graph::unmapAllPointers);
    m.def("read_mesh", [](const std::string& path) { return atcg::IO::read_mesh(path); }, "path"_a);

    m.def("read_pointcloud", [](const std::string& path) { return atcg::IO::read_pointcloud(path); }, "path"_a);

    m.def("read_lines", [](const std::string& path) { return atcg::IO::read_lines(path); }, "path"_a);

    m.def("read_scene", [](const std::string& path) { return atcg::IO::read_scene(path); }, "path"_a);

    m_intrinsics.def(py::init<>())
        .def(py::init<const float, const float, const float, const float, const glm::vec2&>())
        .def(py::init<const glm::mat4&>())
        .def("setAspectRatio", &atcg::CameraIntrinsics::setAspectRatio, "aspect_ratio"_a)
        .def("setFOV", &atcg::CameraIntrinsics::setFOV, "fov"_a)
        .def("setNear", &atcg::CameraIntrinsics::setNear, "near"_a)
        .def("setFar", &atcg::CameraIntrinsics::setFar, "far"_a)
        .def("setOpticalCenter", &atcg::CameraIntrinsics::setOpticalCenter, "optical_center"_a)
        .def("setProjection", &atcg::CameraIntrinsics::setProjection, "projection"_a)
        .def("aspectRatio", &atcg::CameraIntrinsics::aspectRatio)
        .def("FOV", &atcg::CameraIntrinsics::FOV)
        .def("zNear", &atcg::CameraIntrinsics::zNear)
        .def("zFar", &atcg::CameraIntrinsics::zFar)
        .def("opticalCenter", &atcg::CameraIntrinsics::opticalCenter)
        .def("projection", &atcg::CameraIntrinsics::projection);

    m_extrinsics.def(py::init<>())
        .def(py::init<const glm::vec3&, const glm::vec3&>())
        .def(py::init<const glm::mat4&>())
        .def("setPosition", &atcg::CameraExtrinsics::setPosition, "position"_a)
        .def("setTarget", &atcg::CameraExtrinsics::setTarget, "target"_a)
        .def("setExtrinsicMatrix", &atcg::CameraExtrinsics::setExtrinsicMatrix, "view"_a)
        .def("position", &atcg::CameraExtrinsics::position)
        .def("target", &atcg::CameraExtrinsics::target)
        .def("extrinsicMatrix", &atcg::CameraExtrinsics::extrinsicMatrix);


    m_camera
        .def(py::init<atcg::CameraExtrinsics, atcg::CameraIntrinsics>(), "camera_extrinsics"_a, "camera_intrinsics"_a)
        .def("getPosition", &atcg::PerspectiveCamera::getPosition)
        .def("setPosition", &atcg::PerspectiveCamera::setPosition)
        .def("getView", &atcg::PerspectiveCamera::getView)
        .def("setView", &atcg::PerspectiveCamera::setView)
        .def("getProjection", &atcg::PerspectiveCamera::getProjection)
        .def("setProjection", &atcg::PerspectiveCamera::setProjection)
        .def("getLookAt", &atcg::PerspectiveCamera::getLookAt)
        .def("setLookAt", &atcg::PerspectiveCamera::setLookAt)
        .def("getDirection", &atcg::PerspectiveCamera::getDirection)
        .def("getUp", &atcg::PerspectiveCamera::getUp)
        .def("getViewProjection", &atcg::PerspectiveCamera::getViewProjection)
        .def("getAspectRatio", &atcg::PerspectiveCamera::getAspectRatio)
        .def("setAspectRatio", &atcg::PerspectiveCamera::setAspectRatio)
        .def("setFOV", &atcg::PerspectiveCamera::setFOV)
        .def("getFOV", &atcg::PerspectiveCamera::getFOV)
        .def("getNear", &atcg::PerspectiveCamera::getNear)
        .def("setNear", &atcg::PerspectiveCamera::setNear)
        .def("getFar", &atcg::PerspectiveCamera::getFar)
        .def("setFar", &atcg::PerspectiveCamera::setFar);

    m_controller.def(py::init<const atcg::ref_ptr<atcg::PerspectiveCamera>&>(), "camera"_a)
        .def("onUpdate", &atcg::FirstPersonController::onUpdate, "delta_time"_a)
        .def("onEvent", &atcg::FirstPersonController::onEvent, "event"_a)
        .def("getCamera", &atcg::FirstPersonController::getCamera);

    m_serializer.def(py::init<const atcg::ref_ptr<atcg::Scene>&>(), "scene"_a)
        .def("serialize", &atcg::Serialization::SceneSerializer::serialize<>, "file_path"_a)
        .def("deserialize", &atcg::Serialization::SceneSerializer::deserialize<>, "file_path"_a);

    m_performance_panel.def(py::init<>())
        .def(
            "renderPanel",
            [](atcg::PerformancePanel& panel, bool show_window)
            {
                panel.renderPanel(show_window);
                return show_window;
            },
            "show_window"_a)
        .def("registerFrameTime", &atcg::PerformancePanel::registerFrameTime);

    // ------------------- RENDERER ---------------------------------
    m_draw_mode.value("ATCG_DRAW_MODE_TRIANGLE", atcg::DrawMode::ATCG_DRAW_MODE_TRIANGLE)
        .value("ATCG_DRAW_MODE_POINTS", atcg::DrawMode::ATCG_DRAW_MODE_POINTS)
        .value("ATCG_DRAW_MODE_POINTS_SPHERE", atcg::DrawMode::ATCG_DRAW_MODE_POINTS_SPHERE)
        .value("ATCG_DRAW_MODE_EDGES", atcg::DrawMode::ATCG_DRAW_MODE_EDGES)
        .value("ATCG_DRAW_MODE_EDGES_CYLINDER", atcg::DrawMode::ATCG_DRAW_MODE_EDGES_CYLINDER)
        .value("ATCG_DRAW_MODE_INSTANCED", atcg::DrawMode::ATCG_DRAW_MODE_INSTANCED)
        .export_values();

    m_cull_mode.value("ATCG_FRONT_FACE_CULLING", atcg::CullMode::ATCG_FRONT_FACE_CULLING)
        .value("ATCG_BACK_FACE_CULLING", atcg::CullMode::ATCG_BACK_FACE_CULLING)
        .value("ATCG_BOTH_FACE_CULLING", atcg::CullMode::ATCG_BOTH_FACE_CULLING)
        .export_values();

    m_renderer.def("setClearColor", &atcg::Renderer::setClearColor, "color"_a)
        .def("init", &atcg::Renderer::init)
        .def("finishFrame", &atcg::Renderer::finishFrame)
        .def("setClearColor", &atcg::Renderer::setClearColor, "color"_a)
        .def("getClearColor", &atcg::Renderer::getClearColor)
        .def("setPointSize", &atcg::Renderer::setPointSize, "size"_a)
        .def("setLineSize", &atcg::Renderer::setLineSize, "size"_a)
        .def("setViewport", &atcg::Renderer::setViewport, "x"_a, "y"_a, "width"_a, "height"_a)
        .def("setDefaultViewport", &atcg::Renderer::setDefaultViewport)
        .def("useScreenBuffer", &atcg::Renderer::useScreenBuffer)
        .def("clear", &atcg::Renderer::clear)
        .def(
            "draw",
            [](const atcg::ref_ptr<atcg::Graph>& mesh,
               const atcg::ref_ptr<atcg::PerspectiveCamera>& camera,
               const glm::mat4& model,
               const glm::vec3& color,
               const atcg::ref_ptr<atcg::Shader>& shader,
               atcg::DrawMode draw_mode) { atcg::Renderer::draw(mesh, camera, model, color, shader, draw_mode); },
            "graph"_a,
            "camera"_a,
            "model"_a,
            "color"_a,
            "shader"_a,
            "draw_mode"_a)
        .def(
            "drawCADGrid",
            [](const atcg::ref_ptr<atcg::PerspectiveCamera>& camera) { atcg::Renderer::drawCADGrid(camera); },
            "camera"_a)
        .def(
            "drawCameras",
            [](const atcg::ref_ptr<atcg::Scene>& scene, const atcg::ref_ptr<atcg::PerspectiveCamera>& camera)
            { atcg::Renderer::drawCameras(scene, camera); },
            "scene"_a,
            "camera"_a)
        .def("drawCircle", &atcg::Renderer::drawCircle, "position"_a, "radius"_a, "thickness"_a, "color"_a, "camera"_a)
        .def(
            "drawImage",
            [](const atcg::ref_ptr<atcg::Framebuffer>& img) { atcg::Renderer::drawImage(img); },
            "img"_a)
        .def(
            "drawImage",
            [](const atcg::ref_ptr<atcg::Texture2D>& img) { atcg::Renderer::drawImage(img); },
            "img"_a)
        .def("getFramebuffer", &atcg::Renderer::getFramebuffer)
        .def("getEntityIndex", &atcg::Renderer::getEntityIndex, "mouse_pos"_a)
        .def("toggleCulling", &atcg::Renderer::toggleCulling, "enabled"_a)
        .def("toggleMSAA", &atcg::Renderer::toggleMSAA, "enabled"_a)
        .def("screenshot",
             [](const atcg::ref_ptr<atcg::Scene>& scene,
                const atcg::ref_ptr<atcg::PerspectiveCamera>& cam,
                const uint32_t width) { return atcg::Renderer::screenshot(scene, cam, width); })
        .def(
            "screenshot",
            [](const atcg::ref_ptr<atcg::Scene>& scene,
               const atcg::ref_ptr<atcg::PerspectiveCamera>& cam,
               const uint32_t width,
               const std::string& path) { atcg::Renderer::screenshot(scene, cam, width, path); },
            "scene"_a,
            "camera"_a,
            "width"_a,
            "path"_a)
        .def(
            "screenshot",
            [](const atcg::ref_ptr<atcg::Scene>& scene,
               const atcg::ref_ptr<atcg::PerspectiveCamera>& cam,
               const uint32_t width,
               const uint32_t height,
               const std::string& path) { atcg::Renderer::screenshot(scene, cam, width, height, path); },
            "scene"_a,
            "camera"_a,
            "width"_a,
            "height"_a,
            "path"_a)
        .def("resize", &atcg::Renderer::resize)
        .def("getFrame", &atcg::Renderer::getFrame, "device"_a)
        .def("getZBuffer", &atcg::Renderer::getZBuffer, "device"_a)
        .def("toggleDepthTesting", &atcg::Renderer::toggleDepthTesting, "enabled"_a)
        .def("setCullFace", &atcg::Renderer::setCullFace, "mode"_a)
        .def("getFrameCounter", &atcg::Renderer::getFrameCounter)
        .def("popTextureID", &atcg::Renderer::popTextureID)
        .def("pushTextureID", &atcg::Renderer::pushTextureID, "id"_a)
        .def("setMSAA", &atcg::Renderer::setMSAA, "num_samples"_a)
        .def("getMSAA", &atcg::Renderer::getMSAA)
        .def("getFramebufferMSAA", &atcg::Renderer::getFramebufferMSAA);

    m_renderer_system.def(py::init<>())
        .def("setClearColor", &atcg::RendererSystem::setClearColor, "color"_a)
        .def("init", &atcg::RendererSystem::init)
        .def("finishFrame", &atcg::RendererSystem::finishFrame)
        .def("setClearColor", &atcg::RendererSystem::setClearColor, "color"_a)
        .def("getClearColor", &atcg::RendererSystem::getClearColor)
        .def("setPointSize", &atcg::RendererSystem::setPointSize, "size"_a)
        .def("setLineSize", &atcg::RendererSystem::setLineSize, "size"_a)
        .def("setViewport", &atcg::RendererSystem::setViewport, "x"_a, "y"_a, "width"_a, "height"_a)
        .def("setDefaultViewport", &atcg::RendererSystem::setDefaultViewport)
        .def("useScreenBuffer", &atcg::RendererSystem::useScreenBuffer)
        .def("clear", &atcg::RendererSystem::clear)
        .def(
            "draw",
            [](const atcg::ref_ptr<atcg::RendererSystem>& self,
               const atcg::ref_ptr<atcg::Graph>& mesh,
               const atcg::ref_ptr<atcg::PerspectiveCamera>& camera,
               const glm::mat4& model,
               const glm::vec3& color,
               const atcg::ref_ptr<atcg::Shader>& shader,
               atcg::DrawMode draw_mode) { self->draw(mesh, camera, model, color, shader, draw_mode); },
            "graph"_a,
            "camera"_a,
            "model"_a,
            "color"_a,
            "shader"_a,
            "draw_mode"_a)
        .def(
            "drawCADGrid",
            [](const atcg::ref_ptr<atcg::RendererSystem>& self, const atcg::ref_ptr<atcg::PerspectiveCamera>& camera)
            { self->drawCADGrid(camera); },
            "camera"_a)
        .def(
            "drawCameras",
            [](const atcg::ref_ptr<atcg::RendererSystem>& self,
               const atcg::ref_ptr<atcg::Scene>& scene,
               const atcg::ref_ptr<atcg::PerspectiveCamera>& camera) { self->drawCameras(scene, camera); },
            "scene"_a,
            "camera"_a)
        .def("drawCircle",
             &atcg::RendererSystem::drawCircle,
             "position"_a,
             "radius"_a,
             "thickness"_a,
             "color"_a,
             "camera"_a)
        .def(
            "drawImage",
            [](const atcg::ref_ptr<atcg::RendererSystem>& self, const atcg::ref_ptr<atcg::Framebuffer>& img)
            { self->drawImage(img); },
            "img"_a)
        .def(
            "drawImage",
            [](const atcg::ref_ptr<atcg::RendererSystem>& self, const atcg::ref_ptr<atcg::Texture2D>& img)
            { self->drawImage(img); },
            "img"_a)
        .def("getFramebuffer", &atcg::RendererSystem::getFramebuffer)
        .def("getEntityIndex", &atcg::RendererSystem::getEntityIndex, "mouse_pos"_a)
        .def("toggleCulling", &atcg::RendererSystem::toggleCulling, "enabled"_a)
        .def("screenshot",
             [](const atcg::ref_ptr<atcg::RendererSystem>& self,
                const atcg::ref_ptr<atcg::Scene>& scene,
                const atcg::ref_ptr<atcg::PerspectiveCamera>& cam,
                const uint32_t width) { return self->screenshot(scene, cam, width); })
        .def(
            "screenshot",
            [](const atcg::ref_ptr<atcg::RendererSystem>& self,
               const atcg::ref_ptr<atcg::Scene>& scene,
               const atcg::ref_ptr<atcg::PerspectiveCamera>& cam,
               const uint32_t width,
               const std::string& path) { self->screenshot(scene, cam, width, path); },
            "scene"_a,
            "camera"_a,
            "width"_a,
            "path"_a)
        .def(
            "screenshot",
            [](const atcg::ref_ptr<atcg::RendererSystem>& self,
               const atcg::ref_ptr<atcg::Scene>& scene,
               const atcg::ref_ptr<atcg::PerspectiveCamera>& cam,
               const uint32_t width,
               const uint32_t height,
               const std::string& path) { self->screenshot(scene, cam, width, height, path); },
            "scene"_a,
            "camera"_a,
            "width"_a,
            "height"_a,
            "path"_a)
        .def("resize", &atcg::RendererSystem::resize)
        .def("getFrame", &atcg::RendererSystem::getFrame, "device"_a)
        .def("getZBuffer", &atcg::RendererSystem::getZBuffer, "device"_a)
        .def("toggleDepthTesting", &atcg::RendererSystem::toggleDepthTesting, "enabled"_a)
        .def("setCullFace", &atcg::RendererSystem::setCullFace, "mode"_a)
        .def("getFrameCounter", &atcg::RendererSystem::getFrameCounter)
        .def("popTextureID", &atcg::RendererSystem::popTextureID)
        .def("pushTextureID", &atcg::RendererSystem::pushTextureID, "id"_a);

    m_shader.def(py::init<std::string>(), "compute_path"_a)
        .def(py::init<std::string, std::string>(), "vertex_path"_a, "fragment_path"_a)
        .def(py::init<std::string, std::string, std::string>(), "vertex_path"_a, "fragment_path"_a, "geometry_path"_a)
        .def(
            "recompile",
            [](const atcg::ref_ptr<atcg::Shader>& self, const std::string& compute_path)
            { self->recompile(compute_path); },
            "compute_path"_a)
        .def(
            "recompile",
            [](const atcg::ref_ptr<atcg::Shader>& self,
               const std::string& vertex_path,
               const std::string& fragment_path) { self->recompile(vertex_path, fragment_path); },
            "vertex_path"_a,
            "fragment_path"_a)
        .def(
            "recompile",
            [](const atcg::ref_ptr<atcg::Shader>& self,
               const std::string& vertex_path,
               const std::string& fragment_path,
               const std::string& geometry_path) { self->recompile(vertex_path, fragment_path, geometry_path); },
            "vertex_path"_a,
            "fragment_path"_a,
            "geometry_path"_a)
        .def("use", &atcg::Shader::use)
        .def("setInt", &atcg::Shader::setInt, "uniform_name"_a, "value"_a)
        .def("setFloat", &atcg::Shader::setFloat, "uniform_name"_a, "value"_a)
        .def("setVec3", &atcg::Shader::setVec3, "uniform_name"_a, "value"_a)
        .def("setVec4", &atcg::Shader::setVec4, "uniform_name"_a, "value"_a)
        .def("setMat4", &atcg::Shader::setMat4, "uniform_name"_a, "value"_a)
        .def("setMVP", &atcg::Shader::setMVP, "model"_a, "view"_a, "projection"_a)
        .def("dispatch", &atcg::Shader::dispatch, "work_groups"_a)
        .def("hasGeometryShader", &atcg::Shader::hasGeometryShader)
        .def("isComputeShader", &atcg::Shader::isComputeShader)
        .def("getVertexPath", &atcg::Shader::getVertexPath)
        .def("getGeometryPath", &atcg::Shader::getGeometryPath)
        .def("getFragmentPath", &atcg::Shader::getFragmentPath)
        .def("getComputePath", &atcg::Shader::getComputePath);

    m_shader_manager.def("getShader", &atcg::ShaderManager::getShader, "name"_a)
        .def("addShader", &atcg::ShaderManager::addShader, "name"_a, "shader"_a)
        .def("addShaderFromName", &atcg::ShaderManager::addShaderFromName, "name"_a)
        .def("addComputeShaderFromName", &atcg::ShaderManager::addComputeShaderFromName, "name"_a)
        .def("hasShader", &atcg::ShaderManager::hasShader, "name"_a)
        .def("onUpdate", &atcg::ShaderManager::onUpdate);

    m_shader_manager_system.def("getShader", &atcg::ShaderManagerSystem::getShader, "name"_a)
        .def("addShader", &atcg::ShaderManagerSystem::addShader, "name"_a, "shader"_a)
        .def("addShaderFromName", &atcg::ShaderManagerSystem::addShaderFromName, "name"_a)
        .def("addComputeShaderFromName", &atcg::ShaderManagerSystem::addComputeShaderFromName, "name"_a)
        .def("hasShader", &atcg::ShaderManagerSystem::hasShader, "name"_a)
        .def("onUpdate", &atcg::ShaderManagerSystem::onUpdate);

    m_texture_format.value("RG", atcg::TextureFormat::RG)
        .value("RGB", atcg::TextureFormat::RGB)
        .value("RGBA", atcg::TextureFormat::RGBA)
        .value("RGFLOAT", atcg::TextureFormat::RGFLOAT)
        .value("RGBFLOAT", atcg::TextureFormat::RGBFLOAT)
        .value("RGBAFLOAT", atcg::TextureFormat::RGBAFLOAT)
        .value("RINT", atcg::TextureFormat::RINT)
        .value("RINT8", atcg::TextureFormat::RINT8)
        .value("RFLOAT", atcg::TextureFormat::RFLOAT)
        .value("DEPTH", atcg::TextureFormat::DEPTH);

    m_texture_wrap_mode.value("REPEAT", atcg::TextureWrapMode::REPEAT)
        .value("CLAMP_TO_EDGE", atcg::TextureWrapMode::CLAMP_TO_EDGE);

    m_texture_filter_mode.value("NEAREST", atcg::TextureFilterMode::NEAREST)
        .value("LINEAR", atcg::TextureFilterMode::LINEAR);

    m_texture_sampler.def(py::init<>())
        .def(py::init<>(
                 [](atcg::TextureFilterMode filter_mode, atcg::TextureWrapMode wrap_mode)
                 {
                     atcg::TextureSampler sampler;
                     sampler.filter_mode = filter_mode;
                     sampler.wrap_mode   = wrap_mode;
                     return sampler;
                 }),
             py::arg_v("filter_mode", atcg::TextureFilterMode::LINEAR, "linear"),
             py::arg_v("wrap_mode", atcg::TextureWrapMode::REPEAT, "size"))
        .def_readwrite("wrap_mode", &atcg::TextureSampler::wrap_mode)
        .def_readwrite("filter_mode", &atcg::TextureSampler::filter_mode);

    m_texture_specification.def(py::init<>())
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

    m_image.def(py::init<>())
        .def("load", &atcg::Image::load)
        .def("store", &atcg::Image::store)
        .def("applyGamma", &atcg::Image::applyGamma)
        .def("width", &atcg::Image::width)
        .def("height", &atcg::Image::height)
        .def("channels", &atcg::Image::channels)
        .def("name", &atcg::Image::name)
        .def("isHDR", &atcg::Image::isHDR)
        .def_buffer(
            [](const atcg::Image& img) -> py::buffer_info
            {
                bool isHDR  = img.isHDR();
                size_t size = isHDR ? sizeof(float) : sizeof(uint8_t);
                void* data  = img.data().data_ptr();
                return py::buffer_info(data,
                                       size,
                                       isHDR ? py::format_descriptor<float>::format()
                                             : py::format_descriptor<uint8_t>::format(),
                                       3,
                                       {img.height(), img.width(), img.channels()},
                                       {img.width() * img.channels() * size, img.channels() * size, size});
            });

    m.def("imread", &atcg::IO::imread);
    m.def("imwrite", &atcg::IO::imwrite);

    m_texture2d
        .def_static(
            "create",
            [](atcg::TextureSpecification spec) { return atcg::Texture2D::create(spec); },
            "specification"_a)
        .def_static(
            "create",
            [](const atcg::ref_ptr<atcg::Image>& img, atcg::TextureSpecification spec)
            { return atcg::Texture2D::create(img, spec); },
            "img"_a,
            "specification"_a)
        .def_static(
            "create",
            [](const atcg::ref_ptr<atcg::Image>& img) { return atcg::Texture2D::create(img); },
            "img"_a)
        .def_static(
            "create",
            [](const torch::Tensor& img) { return atcg::Texture2D::create(img); },
            "img"_a)
        .def("getID", &atcg::Texture2D::getID)
        .def("use", &atcg::Texture2D::use)
        .def(
            "setData",
            [](const atcg::ref_ptr<atcg::Texture2D>& texture, const torch::Tensor& data) { texture->setData(data); },
            "data"_a)
        .def("__setitem__",
             [](const atcg::ref_ptr<atcg::Texture2D>& texture, py::slice idx, const torch::Tensor& t)
             { texture->setData(t); })
        //.def("setData", [](const atcg::ref_ptr<atcg::Texture2D>& texture, const
        // atcg::ref_ptr<atcg::PixelUnpackBuffer>& data) {texture->setData(data);}, "data"_a)
        .def("getData", &atcg::Texture2D::getData);

    m_texture_cube
        .def_static(
            "create",
            [](atcg::TextureSpecification spec) { return atcg::TextureCube::create(spec); },
            "specification"_a)
        .def_static(
            "create",
            [](const torch::Tensor& img) { return atcg::TextureCube::create(img); },
            "img"_a)
        .def("getID", &atcg::TextureCube::getID)
        .def("use", &atcg::TextureCube::use)
        .def(
            "setData",
            [](const atcg::ref_ptr<atcg::TextureCube>& texture, const torch::Tensor& data) { texture->setData(data); },
            "data"_a)
        //.def("setData", [](const atcg::ref_ptr<atcg::TextureCube>& texture, const
        // atcg::ref_ptr<atcg::PixelUnpackBuffer>& data) {texture->setData(data);}, "data"_a)
        .def("getData", &atcg::TextureCube::getData);

    m_framebuffer.def(py::init<>())
        .def(py::init<uint32_t, uint32_t>())
        .def("use", &atcg::Framebuffer::use)
        .def("attachColor", &atcg::Framebuffer::attachColor)
        .def("attachColorMultiSample", &atcg::Framebuffer::attachColorMultiSample)
        .def("attachTexture", &atcg::Framebuffer::attachTexture)
        .def("attachDepth", [](const atcg::ref_ptr<atcg::Framebuffer>& fbo) { fbo->attachDepth(); })
        .def("attachDepthMultiSample", &atcg::Framebuffer::attachDepthMultiSample)
        .def("blit", &atcg::Framebuffer::blit)
        .def("getColorAttachement", &atcg::Framebuffer::getColorAttachement)
        .def("getDepthAttachement", &atcg::Framebuffer::getDepthAttachement)
        .def("getID", &atcg::Framebuffer::getID)
        .def("width", &atcg::Framebuffer::width)
        .def("height", &atcg::Framebuffer::height)
        .def("currentFramebuffer", &atcg::Framebuffer::currentFramebuffer)
        .def("bindByID", &atcg::Framebuffer::bindByID)
        .def("useDefault", &atcg::Framebuffer::useDefault)
        .def("complete", &atcg::Framebuffer::complete);

    m_vertex_buffer.def(py::init<>())
        .def(py::init<size_t>(), "size"_a)
        .def(py::init(
                 [](py::buffer buffer)
                 {
                     py::buffer_info info = buffer.request();
                     return atcg::make_ref<atcg::VertexBuffer>(info.ptr, info.size * info.itemsize);
                 }),

             "data"_a)
        .def(
            "setData",
            [](const atcg::ref_ptr<atcg::VertexBuffer>& vbo, py::buffer buffer)
            {
                py::buffer_info info = buffer.request();
                vbo->setData(info.ptr, info.size * info.itemsize);
            },
            "data"_a)
        .def("use", &atcg::VertexBuffer::use)
        .def("resize", &atcg::VertexBuffer::resize, "size"_a)
        .def("getLayout", &atcg::VertexBuffer::getLayout)
        .def("setLayout", &atcg::VertexBuffer::setLayout, "setLayout"_a)
        .def("size", &atcg::VertexBuffer::size)
        .def("capacity", &atcg::VertexBuffer::capacity)
        .def("ID", &atcg::VertexBuffer::ID);

    m_buffer_layout.def(py::init<>())
        .def(py::init<const std::vector<atcg::BufferElement>&>(), "elements_a")
        .def("getStride", &atcg::BufferLayout::getStride)
        .def("getElements", &atcg::BufferLayout::getElements);

    m_buffer_element.def(py::init<>())
        .def(py::init<atcg::ShaderDataType, std::string>(), "type"_a, "name"_a)
        .def("getStgetComponentCountride", &atcg::BufferElement::getComponentCount);

    m_shader_data_type.value("None", atcg::ShaderDataType::None)
        .value("Float", atcg::ShaderDataType::Float)
        .value("Float2", atcg::ShaderDataType::Float2)
        .value("Float3", atcg::ShaderDataType::Float3)
        .value("Float4", atcg::ShaderDataType::Float4)
        .value("Mat3", atcg::ShaderDataType::Mat3)
        .value("Mat4", atcg::ShaderDataType::Mat4)
        .value("Int", atcg::ShaderDataType::Int)
        .value("Int2", atcg::ShaderDataType::Int2)
        .value("Int3", atcg::ShaderDataType::Int3)
        .value("Int4", atcg::ShaderDataType::Int4)
        .value("Bool", atcg::ShaderDataType::Bool)
        .export_values();


    // ------------------- Scene ---------------------------------
    m_entity_handle.def(py::init<uint32_t>(), "handle"_a);

    m_material.def(py::init<>())
        .def("getDiffuseTexture", &atcg::Material::getDiffuseTexture)
        .def("getNormalTexture", &atcg::Material::getNormalTexture)
        .def("getRoughnessTexture", &atcg::Material::getRoughnessTexture)
        .def("getMetallicTexture", &atcg::Material::getMetallicTexture)
        .def("setDiffuseTexture", &atcg::Material::setDiffuseTexture)
        .def("setNormalTexture", &atcg::Material::setNormalTexture)
        .def("setRoughnessTexture", &atcg::Material::setRoughnessTexture)
        .def("setMetallicTexture", &atcg::Material::setMetallicTexture)
        .def("setDiffuseColor",
             [](atcg::Material& material, const glm::vec3& color) { material.setDiffuseColor(color); })
        .def("setDiffuseColor",
             [](atcg::Material& material, const glm::vec4& color) { material.setDiffuseColor(color); })
        .def("setRoughness", &atcg::Material::setRoughness)
        .def("setMetallic", &atcg::Material::setMetallic)
        .def("removeNormalMap", &atcg::Material::removeNormalMap);

    m_transform.def(py::init<glm::vec3, glm::vec3, glm::vec3>(), "position"_a, "scale"_a, "rotation"_a)
        .def(py::init<glm::mat4>(), "model"_a)
        .def("setPosition", &atcg::TransformComponent::setPosition, "position"_a)
        .def("setRotation", &atcg::TransformComponent::setRotation, "rotation"_a)
        .def("setScale", &atcg::TransformComponent::setScale, "scale"_a)
        .def("setModel", &atcg::TransformComponent::setModel, "model"_a)
        .def("getPosition", &atcg::TransformComponent::getPosition)
        .def("getRotation", &atcg::TransformComponent::getRotation)
        .def("getScale", &atcg::TransformComponent::getScale)
        .def("getModel", &atcg::TransformComponent::getModel);

    m_geometry.def(py::init<>())
        .def(py::init<const atcg::ref_ptr<atcg::Graph>&>(), "graph"_a)
        .def_readwrite("graph", &atcg::GeometryComponent::graph);

    m_mesh_renderer.def(py::init<>())
        .def(py::init<const atcg::ref_ptr<atcg::Shader>&>(), "shader"_a)
        .def_readwrite("visible", &atcg::MeshRenderComponent::visible)
        .def_readwrite("shader", &atcg::MeshRenderComponent::shader)
        .def_readwrite("material", &atcg::MeshRenderComponent::material);

    m_point_renderer
        .def(py::init<const atcg::ref_ptr<atcg::Shader>&, glm::vec3, float>(), "shader"_a, "color"_a, "point_size"_a)
        .def_readwrite("visible", &atcg::PointRenderComponent::visible)
        .def_readwrite("color", &atcg::PointRenderComponent::color)
        .def_readwrite("shader", &atcg::PointRenderComponent::shader);

    m_point_sphere_renderer.def(py::init<const atcg::ref_ptr<atcg::Shader>&, float>(), "shader"_a, "point_size"_a)
        .def_readwrite("visible", &atcg::PointSphereRenderComponent::visible)
        .def_readwrite("shader", &atcg::PointSphereRenderComponent::shader)
        .def_readwrite("material", &atcg::PointSphereRenderComponent::material);

    m_edge_renderer.def(py::init<glm::vec3>(), "color"_a)
        .def_readwrite("visible", &atcg::EdgeRenderComponent::visible)
        .def_readwrite("color", &atcg::EdgeRenderComponent::color);

    m_edge_cylinder_renderer.def(py::init<float>(), "radius"_a)
        .def_readwrite("visible", &atcg::EdgeCylinderRenderComponent::visible)
        .def_readwrite("material", &atcg::EdgeCylinderRenderComponent::material);

    m_instance_renderer.def(py::init<>())
        .def_readwrite("visible", &atcg::InstanceRenderComponent::visible)
        .def_readwrite("material", &atcg::InstanceRenderComponent::material)
        .def_readwrite("instances", &atcg::InstanceRenderComponent::instance_vbos);

    m_name.def(py::init<>()).def(py::init<std::string>(), "name"_a).def("name", &atcg::NameComponent::name);

    m_point_light.def(py::init<float, glm::vec3>(), "intensity"_a, "color"_a)
        .def_readwrite("intensity", &atcg::PointLightComponent::intensity)
        .def_readwrite("color", &atcg::PointLightComponent::color)
        .def_readwrite("cast_shadow", &atcg::PointLightComponent::cast_shadow);

    m_script_component.def(py::init<>())
        .def(py::init<const atcg::ref_ptr<atcg::PythonScript>&>())
        .def_readwrite("script", &atcg::ScriptComponent::script);

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
            "replaceTransformComponent",
            [](atcg::Entity& entity, atcg::TransformComponent& transform)
            { return entity.replaceComponent<atcg::TransformComponent>(transform); },
            "transform"_a)
        .def(
            "addGeometryComponent",
            [](atcg::Entity& entity, const atcg::ref_ptr<atcg::Graph>& graph)
            { return entity.addComponent<atcg::GeometryComponent>(graph); },
            "graph"_a)
        .def(
            "replaceGeometryComponent",
            [](atcg::Entity& entity, atcg::GeometryComponent& geometry)
            { return entity.replaceComponent<atcg::GeometryComponent>(geometry); },
            "geometry"_a)
        .def(
            "addMeshRenderComponent",
            [](atcg::Entity& entity, const atcg::ref_ptr<atcg::Shader>& shader)
            { return entity.addComponent<atcg::MeshRenderComponent>(shader); },
            "shader"_a)
        .def(
            "replaceMeshRenderComponent",
            [](atcg::Entity& entity, atcg::MeshRenderComponent& component)
            { return entity.replaceComponent<atcg::MeshRenderComponent>(component); },
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
            "replacePointRenderComponent",
            [](atcg::Entity& entity, atcg::PointRenderComponent& component)
            { return entity.replaceComponent<atcg::PointRenderComponent>(component); },
            "component_a")
        .def(
            "addPointSphereRenderComponent",
            [](atcg::Entity& entity, const atcg::ref_ptr<atcg::Shader>& shader, float point_size)
            { return entity.addComponent<atcg::PointSphereRenderComponent>(shader, point_size); },
            "shader"_a,
            "point_size"_a)
        .def(
            "replacePointSphereRenderComponent",
            [](atcg::Entity& entity, atcg::PointSphereRenderComponent& component)
            { return entity.replaceComponent<atcg::PointSphereRenderComponent>(component); },
            "component"_a)
        .def(
            "addEdgeRenderComponent",
            [](atcg::Entity& entity, const glm::vec3& color)
            { return entity.addComponent<atcg::EdgeRenderComponent>(color); },
            "color"_a)
        .def(
            "replaceEdgeRenderComponent",
            [](atcg::Entity& entity, atcg::EdgeRenderComponent& component)
            { return entity.replaceComponent<atcg::EdgeRenderComponent>(component); },
            "component"_a)
        .def(
            "addEdgeCylinderRenderComponent",
            [](atcg::Entity& entity, float radius)
            { return entity.addComponent<atcg::EdgeCylinderRenderComponent>(radius); },
            "radius"_a)
        .def(
            "replaceEdgeCylinderRenderComponent",
            [](atcg::Entity& entity, atcg::EdgeCylinderRenderComponent& component)
            { return entity.replaceComponent<atcg::EdgeCylinderRenderComponent>(component); },
            "component"_a)
        .def("addInstanceRenderComponent",
             [](atcg::Entity& entity) { return entity.addComponent<atcg::InstanceRenderComponent>(); })
        .def(
            "replaceInstanceRenderComponent",
            [](atcg::Entity& entity, atcg::InstanceRenderComponent& component)
            { return entity.replaceComponent<atcg::InstanceRenderComponent>(component); },
            "component"_a)
        .def(
            "addPointLightComponent",
            [](atcg::Entity& entity, float intensity, const glm::vec3& color)
            { return entity.addComponent<atcg::PointLightComponent>(intensity, color); },
            "intensity"_a,
            "color"_a)
        .def(
            "replacePointLightComponent",
            [](atcg::Entity& entity, atcg::PointLightComponent& component)
            { return entity.replaceComponent<atcg::PointLightComponent>(component); },
            "component"_a)
        .def("addNameComponent",
             [](atcg::Entity& entity, const std::string& name)
             { return entity.addComponent<atcg::NameComponent>(name); })
        .def("replaceNameComponent",
             [](atcg::Entity& entity, atcg::NameComponent& component)
             { return entity.replaceComponent<atcg::NameComponent>(component); })
        .def("addScriptComponent",
             [](atcg::Entity& entity, const atcg::ref_ptr<atcg::PythonScript>& script)
             { return entity.addComponent<atcg::ScriptComponent>(script); })
        .def("replaceScriptComponent",
             [](atcg::Entity& entity, atcg::ScriptComponent component)
             { return entity.replaceComponent<atcg::ScriptComponent>(component); })
        .def("hasTransformComponent", &atcg::Entity::hasComponent<atcg::TransformComponent>)
        .def("hasGeometryComponent", &atcg::Entity::hasComponent<atcg::GeometryComponent>)
        .def("hasMeshRenderComponent", &atcg::Entity::hasComponent<atcg::MeshRenderComponent>)
        .def("hasPointRenderComponent", &atcg::Entity::hasComponent<atcg::PointRenderComponent>)
        .def("hasPointSphereRenderComponent", &atcg::Entity::hasComponent<atcg::PointSphereRenderComponent>)
        .def("hasEdgeRenderComponent", &atcg::Entity::hasComponent<atcg::EdgeRenderComponent>)
        .def("hasEdgeCylinderRenderComponent", &atcg::Entity::hasComponent<atcg::EdgeCylinderRenderComponent>)
        .def("hasInstanceRenderComponent", &atcg::Entity::hasComponent<atcg::InstanceRenderComponent>)
        .def("hasPointLightComponent", &atcg::Entity::hasComponent<atcg::PointLightComponent>)
        .def("hasNameComponent", &atcg::Entity::hasComponent<atcg::NameComponent>)
        .def("hasScriptComponent", &atcg::Entity::hasComponent<atcg::ScriptComponent>)
        .def("getTransformComponent", &atcg::Entity::getComponent<atcg::TransformComponent>)
        .def("getGeometryComponent", &atcg::Entity::getComponent<atcg::GeometryComponent>)
        .def("getMeshRenderComponent", &atcg::Entity::getComponent<atcg::MeshRenderComponent>)
        .def("getPointRenderComponent", &atcg::Entity::getComponent<atcg::PointRenderComponent>)
        .def("getPointSphereRenderComponent", &atcg::Entity::getComponent<atcg::PointSphereRenderComponent>)
        .def("getEdgeRenderComponent", &atcg::Entity::getComponent<atcg::EdgeRenderComponent>)
        .def("getPointLightComponent", &atcg::Entity::getComponent<atcg::PointLightComponent>)
        .def("getEdgeCylinderRenderComponent", &atcg::Entity::getComponent<atcg::EdgeCylinderRenderComponent>)
        .def("getInstanceRenderComponent", &atcg::Entity::getComponent<atcg::InstanceRenderComponent>)
        .def("getScriptComponent", &atcg::Entity::getComponent<atcg::ScriptComponent>)
        .def("getNameComponent", &atcg::Entity::getComponent<atcg::NameComponent>);

    m_scene.def(py::init<>([]() { return atcg::make_ref<atcg::Scene>(); }))
        .def(
            "createEntity",
            [](const atcg::ref_ptr<atcg::Scene>& scene, const std::string& name) { return scene->createEntity(name); },
            "name"_a = "Entity")
        .def("getEntityByName", &atcg::Scene::getEntitiesByName, "name"_a)
        .def("getEntities",
             [](const atcg::ref_ptr<atcg::Scene>& scene)
             {
                 std::vector<atcg::Entity> entities;
                 auto view = scene->getAllEntitiesWith<atcg::IDComponent>();
                 for(auto e: view)
                 {
                     entities.push_back(atcg::Entity(e, scene.get()));
                 }
                 return entities;
             })
        .def(
            "getEntityByID",
            [](const atcg::ref_ptr<atcg::Scene>& scene, uint64_t id)
            {
                atcg::UUID uuid(id);
                return scene->getEntityByID(uuid);
            },
            "uuid"_a)
        .def(
            "removeEntity",
            [](const atcg::ref_ptr<atcg::Scene>& scene, uint64_t id)
            {
                atcg::UUID uuid(id);
                scene->removeEntity(uuid);
            },
            "uuid"_a)
        .def(
            "removeEntity",
            [](const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity) { scene->removeEntity(entity); },
            "entity"_a)
        .def("removeAllEntities", &atcg::Scene::removeAllEntites)
        .def("setCamera", &atcg::Scene::setCamera)
        .def("getCamera", &atcg::Scene::getCamera)
        .def("removeCamera", &atcg::Scene::removeCamera)
        .def(
            "setSkybox",
            [](const atcg::ref_ptr<atcg::Scene>& scene, const atcg::ref_ptr<atcg::Image>& skybox)
            { scene->setSkybox(skybox); },
            "skybox"_a)
        .def(
            "setSkybox",
            [](const atcg::ref_ptr<atcg::Scene>& scene, const atcg::ref_ptr<atcg::Texture2D>& skybox)
            { scene->setSkybox(skybox); },
            "skybox"_a)
        .def("hasSkybox", &atcg::Scene::hasSkybox)
        .def("removeSkybox", &atcg::Scene::removeSkybox)
        .def("getSkyboxTexture", &atcg::Scene::getSkyboxTexture)
        .def("getSkyboxCubeMap", &atcg::Scene::getSkyboxCubemap)
        .def(
            "draw",
            [](const atcg::ref_ptr<atcg::Scene>& scene, const atcg::ref_ptr<atcg::PerspectiveCamera>& camera)
            {
                atcg::Dictionary context;
                context.setValue<atcg::ref_ptr<atcg::Camera>>("camera", camera);
                scene->draw(context);
            },
            "camera"_a);

    m_scene_hierarchy_panel.def(py::init<>())
        .def(py::init<const atcg::ref_ptr<atcg::Scene>&>(), "scene"_a)
        .def("renderPanel", &atcg::SceneHierarchyPanel<atcg::ComponentGUIHandler>::renderPanel<>)
        .def("selectEntity", &atcg::SceneHierarchyPanel<atcg::ComponentGUIHandler>::selectEntity, "entity"_a)
        .def("getSelectedEntity", &atcg::SceneHierarchyPanel<atcg::ComponentGUIHandler>::getSelectedEntity);

    m_hit_info.def_readonly("hit", &atcg::Tracing::HitInfo::hit)
        .def_readonly("position", &atcg::Tracing::HitInfo::p)
        .def_readonly("triangle_index", &atcg::Tracing::HitInfo::primitive_idx);

    m.def("prepareAccelerationStructure", &atcg::Tracing::prepareAccelerationStructure);
    m.def("traceRay", atcg::Tracing::traceRay);
    m.def("traceRay",
          [](atcg::Entity entity, py::array_t<float> ray_origins, py::array_t<float> ray_dirs, float t_min, float t_max)
          {
              py::buffer_info origin_info = ray_origins.request();
              py::buffer_info dir_info    = ray_dirs.request();

              auto result_hit      = py::array_t<bool>(origin_info.shape[0]);
              bool* result_hit_ptr = (bool*)result_hit.request().ptr;

              auto result_p           = py::array_t<float>(origin_info.size);
              glm::vec3* result_p_ptr = (glm::vec3*)result_p.request().ptr;

              auto result_idx          = py::array_t<uint32_t>(origin_info.shape[0]);
              uint32_t* result_idx_ptr = (uint32_t*)result_idx.request().ptr;

              uint32_t num_rays = origin_info.shape[0];

              for(int i = 0; i < num_rays; ++i)
              {
                  glm::vec3 o = *((glm::vec3*)origin_info.ptr + i);
                  glm::vec3 d = *((glm::vec3*)dir_info.ptr + i);

                  auto info = atcg::Tracing::traceRay(entity, o, d, t_min, t_max);

                  result_hit_ptr[i] = info.hit;
                  result_p_ptr[i]   = info.p;
                  result_idx_ptr[i] = info.primitive_idx;
              }

              return std::make_tuple(result_hit, result_p, result_idx);
          });

    m_utils.def("AEMap", &atcg::Utils::AEMap, "groundtruth"_a, "prediction"_a, "channel_reduction"_a = "mean");
    m_utils.def("relAEMap",
                &atcg::Utils::relAEMap,
                "groundtruth"_a,
                "prediction"_a,
                "channel_reduction"_a = "mean",
                "delta"_a             = 1e-4f);
    m_utils.def("SEMap", &atcg::Utils::SEMap, "groundtruth"_a, "prediction"_a, "channel_reduction"_a = "mean");
    m_utils.def("relSEMap",
                &atcg::Utils::relSEMap,
                "groundtruth"_a,
                "prediction"_a,
                "channel_reduction"_a = "mean",
                "delta"_a             = 1e-4f);
    m_utils.def("MAE", &atcg::Utils::MAE, "groundtruth"_a, "prediction"_a, "channel_reduction"_a = "mean");
    m_utils.def("relMAE",
                &atcg::Utils::relMAE,
                "groundtruth"_a,
                "prediction"_a,
                "channel_reduction"_a = "mean",
                "delta"_a             = 1e-4f);
    m_utils.def("MSE", &atcg::Utils::MSE, "groundtruth"_a, "prediction"_a, "channel_reduction"_a = "mean");
    m_utils.def("relMSE",
                &atcg::Utils::relMSE,
                "groundtruth"_a,
                "prediction"_a,
                "channel_reduction"_a = "mean",
                "delta"_a             = 1e-4f);

    // ------------------- Network ---------------------------------
    m_tcp_server.def(py::init<>())
        .def("start", &atcg::TCPServer::start)
        .def("stop", &atcg::TCPServer::stop)
        .def("sendToClient", &atcg::TCPServer::sendToClient)
        .def("setOnConnectCallback", &atcg::TCPServer::setOnConnectCallback)
        .def("setOnReceiveCallback", &atcg::TCPServer::setOnReceiveCallback)
        .def("setOnDisconnectCallback", &atcg::TCPServer::setOnDisconnectCallback);

    m_tcp_client.def(py::init<>())
        .def("connect", &atcg::TCPClient::connect)
        .def("disconnect", &atcg::TCPClient::disconnect)
        .def("sendAndWait", &atcg::TCPClient::sendAndWait);

    m_network.def("readByte",
                  [](std::vector<uint8_t>& data, uint32_t& offset)
                  {
                      uint8_t result = atcg::NetworkUtils::readByte(data.data(), offset);
                      return std::make_pair(result, offset);
                  });

    m_network.def("readInt16",
                  [](std::vector<uint8_t>& data, uint32_t& offset)
                  {
                      int16_t result = atcg::NetworkUtils::readInt<int16_t>(data.data(), offset);
                      return std::make_pair(result, offset);
                  });

    m_network.def("readUInt16",
                  [](std::vector<uint8_t>& data, uint32_t& offset)
                  {
                      uint16_t result = atcg::NetworkUtils::readInt<uint16_t>(data.data(), offset);
                      return std::make_pair(result, offset);
                  });

    m_network.def("readInt32",
                  [](std::vector<uint8_t>& data, uint32_t& offset)
                  {
                      int32_t result = atcg::NetworkUtils::readInt<int32_t>(data.data(), offset);
                      return std::make_pair(result, offset);
                  });

    m_network.def("readUInt32",
                  [](std::vector<uint8_t>& data, uint32_t& offset)
                  {
                      uint32_t result = atcg::NetworkUtils::readInt<uint32_t>(data.data(), offset);
                      return std::make_pair(result, offset);
                  });

    m_network.def("readInt64",
                  [](std::vector<uint8_t>& data, uint32_t& offset)
                  {
                      int64_t result = atcg::NetworkUtils::readInt<int64_t>(data.data(), offset);
                      return std::make_pair(result, offset);
                  });

    m_network.def("readUInt64",
                  [](std::vector<uint8_t>& data, uint32_t& offset)
                  {
                      uint64_t result = atcg::NetworkUtils::readInt<uint64_t>(data.data(), offset);
                      return std::make_pair(result, offset);
                  });

    m_network.def("readString",
                  [](std::vector<uint8_t>& data, uint32_t& offset)
                  {
                      auto result = atcg::NetworkUtils::readString(data.data(), offset);
                      return std::make_pair(result, offset);
                  });

    m_network.def("writeByte",
                  [](std::vector<uint8_t>& data, uint32_t& offset, uint8_t toWrite)
                  {
                      atcg::NetworkUtils::writeByte(data.data(), offset, toWrite);
                      return std::make_pair(data, offset);
                  });

    m_network.def("writeInt16",
                  [](std::vector<uint8_t>& data, uint32_t& offset, int16_t toWrite)
                  {
                      atcg::NetworkUtils::writeInt(data.data(), offset, toWrite);
                      return std::make_pair(data, offset);
                  });

    m_network.def("writeUInt16",
                  [](std::vector<uint8_t>& data, uint32_t& offset, uint16_t toWrite)
                  {
                      atcg::NetworkUtils::writeInt(data.data(), offset, toWrite);
                      return std::make_pair(data, offset);
                  });

    m_network.def("writeInt32",
                  [](std::vector<uint8_t>& data, uint32_t& offset, int32_t toWrite)
                  {
                      atcg::NetworkUtils::writeInt(data.data(), offset, toWrite);
                      return std::make_pair(data, offset);
                  });

    m_network.def("writeUInt32",
                  [](std::vector<uint8_t>& data, uint32_t& offset, uint32_t toWrite)
                  {
                      atcg::NetworkUtils::writeInt(data.data(), offset, toWrite);
                      return std::make_pair(data, offset);
                  });

    m_network.def("writeInt64",
                  [](std::vector<uint8_t>& data, uint32_t& offset, int64_t toWrite)
                  {
                      atcg::NetworkUtils::writeInt(data.data(), offset, toWrite);
                      return std::make_pair(data, offset);
                  });

    m_network.def("writeUInt64",
                  [](std::vector<uint8_t>& data, uint32_t& offset, uint64_t toWrite)
                  {
                      atcg::NetworkUtils::writeInt(data.data(), offset, toWrite);
                      return std::make_pair(data, offset);
                  });

    m_network.def("writeBuffer",
                  [](std::vector<uint8_t>& data, uint32_t& offset, std::vector<uint8_t>& toWrite)
                  {
                      atcg::NetworkUtils::writeBuffer(data.data(), offset, toWrite.data(), toWrite.size());
                      return std::make_pair(data, offset);
                  });

    m_network.def("writeString",
                  [](std::vector<uint8_t>& data, uint32_t& offset, const std::string& toWrite)
                  {
                      atcg::NetworkUtils::writeString(data.data(), offset, toWrite);
                      return std::make_pair(data, offset);
                  });

    // ------------------- Scripting ---------------------------------
    m_scriptengine.def(py::init<>())
        .def("init", &atcg::PythonScriptEngine::init)
        .def("destroy", &atcg::PythonScriptEngine::destroy);

    m_script.def(py::init<const std::filesystem::path&>())
        .def("init", &atcg::PythonScript::init)
        .def("onAttach", &atcg::PythonScript::onAttach)
        .def("onUpdate", &atcg::PythonScript::onUpdate)
        .def("onEvent", &atcg::PythonScript::onEvent)
        .def("onDetach", &atcg::PythonScript::onDetach)
        .def("reload", &atcg::PythonScript::reload);

    m.def("handleScriptReloads", &atcg::Scripting::handleScriptReloads);
    m.def("handleScriptEvents", &atcg::Scripting::handleScriptEvents);
    m.def("handleScriptUpdates", &atcg::Scripting::handleScriptUpdates);

    // IMGUI BINDINGS

#ifndef ATCG_HEADLESS
    m_imgui.def("BeginMainMenuBar", &ImGui::BeginMainMenuBar);
    m_imgui.def("EndMainMenuBar", &ImGui::EndMainMenuBar);
    m_imgui.def("BeginMenu", &ImGui::BeginMenu, py::arg("label"), py::arg("enabled") = true);
    m_imgui.def("EndMenu", &ImGui::EndMenu);
    m_imgui.def(
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
    m_imgui.def(
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
    m_imgui.def("End", &ImGui::End);
    m_imgui.def(
        "Checkbox",
        [](const char* label, bool* v)
        {
            auto ret = ImGui::Checkbox(label, v);
            return std::make_tuple(ret, v);
        },
        py::arg("label"),
        py::arg("v"),
        py::return_value_policy::automatic_reference);
    m_imgui.def(
        "Button",
        [](const char* label)
        {
            auto ret = ImGui::Button(label);
            return ret;
        },
        py::arg("label"),
        py::return_value_policy::automatic_reference);
    m_imgui.def(
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
    m_imgui.def(
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
    m_imgui.def(
        "Text",
        [](const char* fmt)
        {
            ImGui::Text(fmt);
            return;
        },
        py::arg("fmt"),
        py::return_value_policy::automatic_reference);

    m_imgui.def(
        "Image",
        [](uint32_t textureID, uint32_t width, uint32_t height)
        {
            ImGui::Image((ImTextureID)textureID, ImVec2(width, height), ImVec2 {0, 1}, ImVec2 {1, 0});
            return;
        },
        py::arg("textureID"),
        py::arg("width"),
        py::arg("height"),
        py::return_value_policy::automatic_reference);

    m_imgui.def(
        "plot",
        [](std::vector<std::vector<float>>& data, const std::vector<std::string>& names, const float line_width = -1)
        {
            if(ImPlot::BeginPlot("##", ImVec2(-1, -1)))
            {
                ImPlot::SetupAxes("X", "Y", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                ImPlot::SetupLegend(ImPlotLocation_North, ImPlotLegendFlags_Horizontal | ImPlotLegendFlags_Outside);
                for(int i = 0; i < data.size(); ++i)
                {
                    std::vector<float> x(data[i].size());
                    std::iota(x.begin(), x.end(), 0.0f);

                    ImPlot::SetNextLineStyle(IMPLOT_AUTO_COL, line_width);
                    ImPlot::PlotLine(names[i].c_str(), x.data(), data[i].data(), data[i].size(), 0, 0, sizeof(float));
                }
                ImPlot::EndPlot();
            }
        });

    m_imgui.def("isUsing", &ImGuizmo::IsUsing);

    m_guizmo_operation.value("TRANSLATE", ImGuizmo::OPERATION::TRANSLATE)
        .value("ROTATE", ImGuizmo::OPERATION::ROTATE)
        .value("SCALE", ImGuizmo::OPERATION::SCALE)
        .export_values();
    m_imgui.def("drawGuizmo", atcg::drawGuizmo);
#endif

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}