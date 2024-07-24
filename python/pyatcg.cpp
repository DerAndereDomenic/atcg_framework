#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>
#include <Core/EntryPoint.h>
#include <ATCG.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <torch/python.h>

#include <imgui.h>

#include "pyatcg.h"

#define STRINGIFY(x)       #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


//* This function isn't called but is needed for the linker
atcg::Application* atcg::createApplication()
{
    return nullptr;
}

int python_main(atcg::Application* app)
{
    return atcg::atcg_main(app);
}

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_DECLARE_HOLDER_TYPE(T, atcg::ref_ptr<T>);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = R"pbdoc(
        Pybind11 atcg plugin
        -----------------------
        .. currentmodule:: pyatcg
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    // ---------------- CORE ---------------------
    ATCG_DEFINE_MODULES(m)


    m_window_props.def(py::init<>([]() { return atcg::WindowProps(); }))
        .def(py::init<const std::string&, uint32_t, uint32_t, int32_t, int32_t, bool>())
        .def_readwrite("tile", &atcg::WindowProps::title)
        .def_readwrite("width", &atcg::WindowProps::width)
        .def_readwrite("height", &atcg::WindowProps::height)
        .def_readwrite("pos_x", &atcg::WindowProps::pos_x)
        .def_readwrite("pos_y", &atcg::WindowProps::pos_y)
        .def_readwrite("vsync", &atcg::WindowProps::vsync)
        .def_readwrite("hidden", &atcg::WindowProps::hidden);

    m.def("start", &python_main, py::arg("application"));
    m.def("init",
          []()
          {
              atcg::WindowProps props;
              props.hidden                           = true;
              std::unique_ptr<PythonApplication> app = std::make_unique<PythonApplication>(props);
              return app;
          });
    m.def("print_statistics", &atcg::print_statistics);
    m_application.def(py::init<atcg::Layer*>()).def(py::init<atcg::Layer*, atcg::WindowProps>());
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
        .def_buffer(
            [](glm::mat3& M) -> py::buffer_info
            {
                float data[3][3];

                for(int i = 0; i < 3; ++i)
                {
                    for(int j = 0; j < 3; ++j)
                    {
                        data[i][j] = M[j][i];
                    }
                }

                return py::buffer_info(data,
                                       sizeof(float),
                                       py::format_descriptor<float>::format(),
                                       2,
                                       {3, 3},
                                       {sizeof(float) * 3, sizeof(float)});
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
        .def_buffer(
            [](glm::mat4& M) -> py::buffer_info
            {
                float data[4][4];

                for(int i = 0; i < 4; ++i)
                {
                    for(int j = 0; j < 4; ++j)
                    {
                        data[i][j] = M[j][i];
                    }
                }

                return py::buffer_info(data,
                                       sizeof(float),
                                       py::format_descriptor<float>::format(),
                                       2,
                                       {4, 4},
                                       {sizeof(float) * 4, sizeof(float)});
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
        .def_readonly_static("TANGNET_BEGIN", &atcg::VertexSpecification::TANGNET_BEGIN)
        .def_readonly_static("TANGNET_END", &atcg::VertexSpecification::TANGNET_END)
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


    m_camera
        .def(py::init<>([](float aspect_ratio) { return atcg::make_ref<atcg::PerspectiveCamera>(aspect_ratio); }),
             "aspect_ratio"_a)
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

    m_controller
        .def(py::init<>([](float aspect_ratio) { return atcg::make_ref<atcg::FirstPersonController>(aspect_ratio); }))
        .def("onUpdate", &atcg::FirstPersonController::onUpdate, "delta_time"_a)
        .def("onEvent", &atcg::FirstPersonController::onEvent, "event"_a)
        .def("getCamera", &atcg::FirstPersonController::getCamera);

    m_serializer.def(py::init<const atcg::ref_ptr<atcg::Scene>&>(), "scene"_a)
        .def("serialize", &atcg::Serializer<atcg::ComponentSerializer>::serialize<>, "file_path"_a)
        .def("deserialize", &atcg::Serializer<atcg::ComponentSerializer>::deserialize<>, "file_path"_a);

    // ------------------- RENDERER ---------------------------------
    m_renderer.def_static("setClearColor", &atcg::Renderer::setClearColor, "color"_a)
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
        .def_static("toggleCulling", &atcg::Renderer::toggleCulling, "enabled"_a)
        .def_static("screenshot",
                    [](const atcg::ref_ptr<atcg::Scene>& scene,
                       const atcg::ref_ptr<atcg::PerspectiveCamera>& cam,
                       const uint32_t width) { return atcg::Renderer::screenshot(scene, cam, width); })
        .def_static(
            "screenshot",
            [](const atcg::ref_ptr<atcg::Scene>& scene,
               const atcg::ref_ptr<atcg::PerspectiveCamera>& cam,
               const uint32_t width,
               const std::string& path) { atcg::Renderer::screenshot(scene, cam, width, path); },
            "scene"_a,
            "camera"_a,
            "width"_a,
            "path"_a);

    m_shader.def(py::init<std::string, std::string>(), "vertex_path"_a, "fragment_path"_a)
        .def(py::init<std::string, std::string, std::string>(), "vertex_path"_a, "fragment_path"_a, "geometry_path"_a)
        .def("use", &atcg::Shader::use)
        .def("setInt", &atcg::Shader::setInt, "uniform_name"_a, "value"_a)
        .def("setFloat", &atcg::Shader::setFloat, "uniform_name"_a, "value"_a)
        .def("setVec3", &atcg::Shader::setVec3, "uniform_name"_a, "value"_a)
        .def("setVec4", &atcg::Shader::setVec4, "uniform_name"_a, "value"_a)
        .def("setMat4", &atcg::Shader::setMat4, "uniform_name"_a, "value"_a)
        .def("setMVP", &atcg::Shader::setMVP, "model"_a, "view"_a, "projection"_a);

    m_shader_manager.def_static("getShader", &atcg::ShaderManager::getShader, "name"_a)
        .def_static("addShader", &atcg::ShaderManager::addShader, "name"_a, "shader"_a)
        .def_static("addShaderFromName", &atcg::ShaderManager::addShaderFromName, "name"_a);

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
        .def("getID", &atcg::Texture2D::getID)
        .def("setData", &atcg::Texture2D::setData, "data"_a)
        .def("getData", &atcg::Texture2D::getData);

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

    m_name.def(py::init<>()).def(py::init<std::string>(), "name"_a).def_readwrite("name", &atcg::NameComponent::name);

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
        .def("addNameComponent",
             [](atcg::Entity& entity, const std::string& name)
             { return entity.addComponent<atcg::NameComponent>(name); })
        .def("replaceNameComponent",
             [](atcg::Entity& entity, atcg::NameComponent& component)
             { return entity.replaceComponent<atcg::NameComponent>(component); })
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
                 for(auto e: view)
                 {
                     entities.push_back(atcg::Entity(e, scene.get()));
                 }
                 return entities;
             });

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

    // IMGUI BINDINGS

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
            ImGui::Image((void*)(uint64_t)(textureID), ImVec2(width, height), ImVec2 {0, 1}, ImVec2 {1, 0});
            return;
        },
        py::arg("textureID"),
        py::arg("width"),
        py::arg("height"),
        py::return_value_policy::automatic_reference);

    m_imgui.def("isUsing", &ImGuizmo::IsUsing);

    m_guizmo_operation.value("TRANSLATE", ImGuizmo::OPERATION::TRANSLATE)
        .value("ROTATE", ImGuizmo::OPERATION::ROTATE)
        .value("SCALE", ImGuizmo::OPERATION::SCALE)
        .export_values();
    m_imgui.def("drawGuizmo", atcg::drawGuizmo);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}