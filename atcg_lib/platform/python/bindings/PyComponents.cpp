#include "Bindings.h"

using namespace pybind11::literals;

namespace atcg
{
namespace python
{
// Scene components
void bindComponents(BindingContext& ctx)
{
    py::module_ m = ctx.module();

    auto m_transform              = py::class_<atcg::TransformComponent>(m, "TransformComponent");
    auto m_geometry               = py::class_<atcg::GeometryComponent>(m, "GeometryComponent");
    auto m_mesh_renderer          = py::class_<atcg::MeshRenderComponent>(m, "MeshRenderComponent");
    auto m_point_renderer         = py::class_<atcg::PointRenderComponent>(m, "PointRenderComponent");
    auto m_point_sphere_renderer  = py::class_<atcg::PointSphereRenderComponent>(m, "PointSphereRenderComponent");
    auto m_edge_renderer          = py::class_<atcg::EdgeRenderComponent>(m, "EdgeRenderComponent");
    auto m_edge_cylinder_renderer = py::class_<atcg::EdgeCylinderRenderComponent>(m, "EdgeCylinderRenderComponent");
    auto m_instance_renderer      = py::class_<atcg::InstanceRenderComponent>(m, "InstanceRenderComponent");
    auto m_camera_component       = py::class_<atcg::CameraComponent>(m, "CameraComponent");

    auto m_name                 = py::class_<atcg::NameComponent>(m, "NameComponent");
    auto m_point_light          = py::class_<atcg::PointLightComponent>(m, "PointLightComponent");
    auto m_script_component     = py::class_<atcg::ScriptComponent>(m, "ScriptComponent");
    auto m_mesh_light_component = py::class_<atcg::MeshLightComponent>(m, "MeshLightComponent");

    ctx.define(
        [=]() mutable
        {
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
                .def(py::init<atcg::AssetHandle>(), "graph"_a)
                .def("graph", &atcg::GeometryComponent::graph)
                .def_readwrite("graph_handle", &atcg::GeometryComponent::graph_handle);

            m_mesh_renderer.def(py::init<>())
                .def(py::init<const atcg::ref_ptr<atcg::Shader>&>(), "shader"_a)
                .def_readwrite("visible", &atcg::MeshRenderComponent::visible)
                .def("shader", &atcg::MeshRenderComponent::shader)
                .def("material", &atcg::MeshRenderComponent::material)
                .def_readwrite("material_handle", &atcg::MeshRenderComponent::material_handle)
                .def_readwrite("shader_handle", &atcg::MeshRenderComponent::shader_handle)
                .def_readwrite("cull_mode", &atcg::MeshRenderComponent::cull_mode);

            m_point_renderer
                .def(py::init<const atcg::ref_ptr<atcg::Shader>&, glm::vec3, float>(),
                     "shader"_a,
                     "color"_a,
                     "point_size"_a)
                .def_readwrite("visible", &atcg::PointRenderComponent::visible)
                .def_readwrite("color", &atcg::PointRenderComponent::color)
                .def("shader", &atcg::PointRenderComponent::shader)
                .def_readwrite("shader_handle", &atcg::PointRenderComponent::shader_handle);

            m_point_sphere_renderer
                .def(py::init<const atcg::ref_ptr<atcg::Shader>&, float>(), "shader"_a, "point_size"_a)
                .def_readwrite("visible", &atcg::PointSphereRenderComponent::visible)
                .def("shader", &atcg::PointSphereRenderComponent::shader)
                .def("material", &atcg::PointSphereRenderComponent::material)
                .def_readwrite("material_handle", &atcg::PointSphereRenderComponent::material_handle)
                .def_readwrite("shader_handle", &atcg::PointSphereRenderComponent::shader_handle)
                .def_readwrite("cull_mode", &atcg::PointSphereRenderComponent::cull_mode);

            m_edge_renderer.def(py::init<glm::vec3>(), "color"_a)
                .def_readwrite("visible", &atcg::EdgeRenderComponent::visible)
                .def_readwrite("color", &atcg::EdgeRenderComponent::color);

            m_edge_cylinder_renderer.def(py::init<float>(), "radius"_a)
                .def_readwrite("visible", &atcg::EdgeCylinderRenderComponent::visible)
                .def("material", &atcg::EdgeCylinderRenderComponent::material)
                .def_readwrite("material_handle", &atcg::EdgeCylinderRenderComponent::material_handle)
                .def_readwrite("cull_mode", &atcg::EdgeCylinderRenderComponent::cull_mode);

            m_instance_renderer.def(py::init<>())
                .def_readwrite("visible", &atcg::InstanceRenderComponent::visible)
                .def("material", &atcg::InstanceRenderComponent::material)
                .def_readwrite("instances", &atcg::InstanceRenderComponent::instance_vbos)
                .def_readwrite("material_handle", &atcg::InstanceRenderComponent::material_handle);

            m_camera_component.def(py::init<>())
                .def(py::init<>([](const atcg::ref_ptr<atcg::PerspectiveCamera>& camera,
                                   const uint32_t width,
                                   const uint32_t height) { return atcg::CameraComponent(camera, width, height); }),
                     "camera"_a,
                     "width"_a,
                     "height"_a)
                .def("camera",
                     [](const atcg::CameraComponent& self)
                     { return std::dynamic_pointer_cast<atcg::PerspectiveCamera>(self.camera); })
                .def_readwrite("color", &atcg::CameraComponent::color)
                .def_readwrite("width", &atcg::CameraComponent::width)
                .def_readwrite("height", &atcg::CameraComponent::height)
                .def_readwrite("preview", &atcg::CameraComponent::preview)
                .def("image", &atcg::CameraComponent::image);

            m_name.def(py::init<>()).def(py::init<std::string>(), "name"_a).def("name", &atcg::NameComponent::name);

            m_point_light.def(py::init<float, glm::vec3>(), "intensity"_a, "color"_a)
                .def_readwrite("intensity", &atcg::PointLightComponent::intensity)
                .def_readwrite("color", &atcg::PointLightComponent::color)
                .def_readwrite("cast_shadow", &atcg::PointLightComponent::cast_shadow);

            m_script_component.def(py::init<>())
                .def(py::init<const atcg::ref_ptr<atcg::PythonScript>&>())
                .def("script", &atcg::ScriptComponent::script)
                .def_readwrite("script_handle", &atcg::ScriptComponent::script_handle);

            m_mesh_light_component.def(py::init<>())
                .def("setEmissiveColor", &atcg::MeshLightComponent::setEmissiveColor, "color"_a)
                .def("getEmissiveTexture", &atcg::MeshLightComponent::getEmissiveTexture)
                .def_readwrite("intensity", &atcg::MeshLightComponent::intensity);
        });
}
}    // namespace python
}    // namespace atcg
