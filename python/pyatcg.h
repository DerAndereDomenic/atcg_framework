#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <torch/python.h>
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

#define ATCG_DEFINE_MODULES(m)                                                                                                  \
    py::class_<atcg::Application>(m, "Application");                                                                            \
    auto m_application = py::class_<PythonApplication, atcg::Application>(m, "PythonApplication");                              \
    auto m_layer       = py::class_<atcg::Layer, PythonLayer, std::unique_ptr<atcg::Layer, py::nodelete>>(m, "Layer");          \
    auto m_event       = py::class_<atcg::Event>(m, "Event");                                                                   \
    auto m_camera =                                                                                                             \
        py::class_<atcg::PerspectiveCamera, atcg::ref_ptr<atcg::PerspectiveCamera>>(m, "PerspectiveCamera");                    \
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
    auto m_serializer            = py::class_<atcg::Serializer<atcg::ComponentSerializer>>(m, "Serializer");                    \
    auto m_renderer              = py::class_<atcg::Renderer>(m, "Renderer");                                                   \
    auto m_shader                = py::class_<atcg::Shader, atcg::ref_ptr<atcg::Shader>>(m, "Shader");                          \
    auto m_shader_manager        = py::class_<atcg::ShaderManager>(m, "ShaderManager");                                         \
    auto m_texture_format        = py::enum_<atcg::TextureFormat>(m, "TextureFormat");                                          \
    auto m_texture_wrap_mode     = py::enum_<atcg::TextureWrapMode>(m, "TextureWrapMode");                                      \
    auto m_texture_filter_mode   = py::enum_<atcg::TextureFilterMode>(m, "TextureFilterMode");                                  \
    auto m_texture_sampler       = py::class_<atcg::TextureSampler>(m, "TextureSampler");                                       \
    auto m_texture_specification = py::class_<atcg::TextureSpecification>(m, "TextureSpecification");                           \
    auto m_image          = py::class_<atcg::Image, atcg::ref_ptr<atcg::Image>>(m, "Image", py::buffer_protocol());             \
    auto m_texture2d      = py::class_<atcg::Texture2D, atcg::ref_ptr<atcg::Texture2D>>(m, "Texture2D");                        \
    auto m_entity_handle  = py::class_<entt::entity>(m, "EntityHandle");                                                        \
    auto m_material       = py::class_<atcg::Material>(m, "Material");                                                          \
    auto m_transform      = py::class_<atcg::TransformComponent>(m, "TransformComponent");                                      \
    auto m_geometry       = py::class_<atcg::GeometryComponent>(m, "GeometryComponent");                                        \
    auto m_mesh_renderer  = py::class_<atcg::MeshRenderComponent>(m, "MeshRenderComponent");                                    \
    auto m_point_renderer = py::class_<atcg::PointRenderComponent>(m, "PointRenderComponent");                                  \
    auto m_point_sphere_renderer  = py::class_<atcg::PointSphereRenderComponent>(m, "PointSphereRenderComponent");              \
    auto m_edge_renderer          = py::class_<atcg::EdgeRenderComponent>(m, "EdgeRenderComponent");                            \
    auto m_edge_cylinder_renderer = py::class_<atcg::EdgeCylinderRenderComponent>(m, "EdgeCylinderRenderComponent");            \
    auto m_name                   = py::class_<atcg::NameComponent>(m, "NameComponent");                                        \
    auto m_scene_hierarchy_panel =                                                                                              \
        py::class_<atcg::SceneHierarchyPanel<atcg::ComponentGUIHandler>>(m, "SceneHierarchyPanel");                             \
    auto m_hit_info         = py::class_<atcg::Tracing::HitInfo>(m, "HitInfo");                                                 \
    auto m_utils            = m.def_submodule("Utils");                                                                         \
    auto m_imgui            = m.def_submodule("ImGui");                                                                         \
    auto m_guizmo_operation = py::enum_<ImGuizmo::OPERATION>(m_imgui, "GuizmoOperation");