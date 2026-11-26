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
    auto m_id                   = py::class_<atcg::IDComponent>(m, "IDComponent");
    auto m_point_light          = py::class_<atcg::PointLightComponent>(m, "PointLightComponent");
    auto m_script_component     = py::class_<atcg::ScriptComponent>(m, "ScriptComponent");
    auto m_mesh_light_component = py::class_<atcg::MeshLightComponent>(m, "MeshLightComponent");

    auto m_transparency         = py::class_<atcg::TransparencyComponent>(m, "TransparencyComponent");
    auto m_homogeneous_medium   = py::class_<atcg::HomogeneousMediumComponent>(m, "HomogeneousMediumComponent");
    auto m_heterogeneous_medium = py::class_<atcg::HeterogeneousMediumComponent>(m, "HeterogeneousMediumComponent");
    auto m_grid_component =
        py::class_<atcg::HeterogeneousMediumComponent::GridComponent>(m_heterogeneous_medium, "GridComponent");

    ctx.define(
        [=]() mutable
        {
            m_transform
                .def(py::init<const glm::vec3&, const glm::vec3&, const glm::vec3&>(),
                     "position"_a = glm::vec3(0),
                     "scale"_a    = glm::vec3(1),
                     "rotation"_a = glm::vec3(0))
                .def(py::init<const glm::mat4&>(), "model"_a)
                .def("setPosition", &atcg::TransformComponent::setPosition, "position"_a)
                .def("setRotation", &atcg::TransformComponent::setRotation, "rotation"_a)
                .def("setScale", &atcg::TransformComponent::setScale, "scale"_a)
                .def("setModel", &atcg::TransformComponent::setModel, "model"_a)
                .def("getPosition", &atcg::TransformComponent::getPosition)
                .def("getRotation", &atcg::TransformComponent::getRotation)
                .def("getScale", &atcg::TransformComponent::getScale)
                .def("getModel", &atcg::TransformComponent::getModel)
                .def("revision", &atcg::TransformComponent::revision);

            m_geometry.def(py::init<>())
                .def(py::init<const atcg::ref_ptr<atcg::Graph>&>(), "graph"_a)
                .def(py::init<atcg::AssetHandle>(), "graph"_a)
                .def("graph", &atcg::GeometryComponent::graph)
                .def_readwrite("graph_handle", &atcg::GeometryComponent::graph_handle)
                .def_readwrite("draw_bounding_box", &atcg::GeometryComponent::draw_bounding_box);

            m_mesh_renderer.def(py::init<>())
                .def(py::init<const atcg::ref_ptr<atcg::Shader>&>(), "shader"_a)
                .def_readwrite("visible", &atcg::MeshRenderComponent::visible)
                .def("shader", &atcg::MeshRenderComponent::shader)
                .def("material", &atcg::MeshRenderComponent::material)
                .def_readwrite("material_handle", &atcg::MeshRenderComponent::material_handle)
                .def_readwrite("shader_handle", &atcg::MeshRenderComponent::shader_handle)
                .def_readwrite("cull_mode", &atcg::MeshRenderComponent::cull_mode)
                .def_readwrite("receive_shadow", &atcg::MeshRenderComponent::receive_shadow)
                .def_readwrite("default_shader", &atcg::MeshRenderComponent::default_shader)
                .def_readwrite("default_material", &atcg::MeshRenderComponent::default_material);

            m_point_renderer.def(py::init<const glm::vec3&, float>(), "color"_a = glm::vec3(1), "point_size"_a = 1.0f)
                .def(py::init<const atcg::ref_ptr<atcg::Shader>&, const glm::vec3&, float>(),
                     "shader"_a,
                     "color"_a      = glm::vec3(1),
                     "point_size"_a = 1.0f)
                .def_readwrite("visible", &atcg::PointRenderComponent::visible)
                .def_readwrite("color", &atcg::PointRenderComponent::color)
                .def_readwrite("point_size", &atcg::PointRenderComponent::point_size)
                .def("shader", &atcg::PointRenderComponent::shader)
                .def_readwrite("shader_handle", &atcg::PointRenderComponent::shader_handle)
                .def_readwrite("default_shader", &atcg::PointRenderComponent::default_shader);

            m_point_sphere_renderer.def(py::init<float>(), "point_size"_a = 0.1f)
                .def(py::init<const atcg::ref_ptr<atcg::Shader>&, float>(), "shader"_a, "point_size"_a = 0.1f)
                .def_readwrite("visible", &atcg::PointSphereRenderComponent::visible)
                .def("shader", &atcg::PointSphereRenderComponent::shader)
                .def("material", &atcg::PointSphereRenderComponent::material)
                .def_readwrite("material_handle", &atcg::PointSphereRenderComponent::material_handle)
                .def_readwrite("shader_handle", &atcg::PointSphereRenderComponent::shader_handle)
                .def_readwrite("cull_mode", &atcg::PointSphereRenderComponent::cull_mode)
                .def_readwrite("point_size", &atcg::PointSphereRenderComponent::point_size)
                .def_readwrite("default_shader", &atcg::PointSphereRenderComponent::default_shader);

            m_edge_renderer.def(py::init<const glm::vec3&>(), "color"_a = glm::vec3(1))
                .def_readwrite("visible", &atcg::EdgeRenderComponent::visible)
                .def_readwrite("color", &atcg::EdgeRenderComponent::color);

            m_edge_cylinder_renderer.def(py::init<float>(), "radius"_a = 0.001f)
                .def_readwrite("visible", &atcg::EdgeCylinderRenderComponent::visible)
                .def_readwrite("radius", &atcg::EdgeCylinderRenderComponent::radius)
                .def("material", &atcg::EdgeCylinderRenderComponent::material)
                .def_readwrite("material_handle", &atcg::EdgeCylinderRenderComponent::material_handle)
                .def_readwrite("cull_mode", &atcg::EdgeCylinderRenderComponent::cull_mode);

            m_instance_renderer.def(py::init<>())
                .def_readwrite("visible", &atcg::InstanceRenderComponent::visible)
                .def("material", &atcg::InstanceRenderComponent::material)
                .def("shader", &atcg::InstanceRenderComponent::shader)
                .def("addInstanceBuffer", &atcg::InstanceRenderComponent::addInstanceBuffer, "instance_vbo"_a)
                .def_readwrite("instances",
                               &atcg::InstanceRenderComponent::instance_vbos,
                               "The instance vertex buffers. Reading returns a copy of the list, so in-place changes "
                               "such as append() are lost. Assign a new list or use addInstanceBuffer() instead.")
                .def_readwrite("material_handle", &atcg::InstanceRenderComponent::material_handle)
                .def_readwrite("shader_handle", &atcg::InstanceRenderComponent::shader_handle)
                .def_readwrite("receive_shadow", &atcg::InstanceRenderComponent::receive_shadow)
                .def_readwrite("default_shader", &atcg::InstanceRenderComponent::default_shader);

            m_camera_component.def(py::init<>())
                .def(py::init<const atcg::ref_ptr<atcg::Camera>&, uint32_t, uint32_t>(),
                     "camera"_a,
                     "width"_a  = 1024,
                     "height"_a = 1024)
                .def("camera", [](const atcg::CameraComponent& self) { return self.camera; })
                .def(
                    "setCamera",
                    [](atcg::CameraComponent& self, const atcg::ref_ptr<atcg::Camera>& camera)
                    {
                        self.camera      = camera;
                        self.perspective = dynamic_cast<atcg::PerspectiveCamera*>(camera.get()) != nullptr;
                    },
                    "camera"_a)
                .def_readwrite("color", &atcg::CameraComponent::color)
                .def_readwrite("perspective", &atcg::CameraComponent::perspective)
                .def_readwrite("width", &atcg::CameraComponent::width)
                .def_readwrite("height", &atcg::CameraComponent::height)
                .def_readwrite("render_scale", &atcg::CameraComponent::render_scale)
                .def_readwrite("preview", &atcg::CameraComponent::preview)
                .def_readwrite("image_handle", &atcg::CameraComponent::image_handle)
                .def_readwrite("render_preview", &atcg::CameraComponent::render_preview)
                .def("image", &atcg::CameraComponent::image);

            m_name.def(py::init<>()).def(py::init<std::string>(), "name"_a).def("name", &atcg::NameComponent::name);

            m_id.def(py::init<>())
                .def(py::init<uint64_t>(), "id"_a)
                .def("ID", [](const atcg::IDComponent& self) { return (uint64_t)self.ID(); })
                .def("__int__", [](const atcg::IDComponent& self) { return (uint64_t)self.ID(); });

            m_point_light.def(py::init<float, const glm::vec3&>(), "intensity"_a = 1.0f, "color"_a = glm::vec3(1))
                .def_readwrite("intensity", &atcg::PointLightComponent::intensity)
                .def_readwrite("color", &atcg::PointLightComponent::color)
                .def_readwrite("cast_shadow", &atcg::PointLightComponent::cast_shadow);

            m_script_component.def(py::init<>())
                .def(py::init<const atcg::ref_ptr<atcg::Script>&>(), "script"_a)
                .def("script", &atcg::ScriptComponent::script)
                .def(
                    "setScript",
                    [](atcg::ScriptComponent& self, const atcg::ref_ptr<atcg::Script>& script)
                    {
                        if(!script)
                        {
                            ATCG_WARN("ScriptComponent.setScript: script is None");
                            return;
                        }
                        self.setScript(script);
                    },
                    "script"_a)
                .def("behavior", &atcg::ScriptComponent::behavior, "scene"_a, "entity"_a, "recreate"_a = false)
                .def("resetBehavior", &atcg::ScriptComponent::resetBehavior)
                .def_readwrite("script_handle", &atcg::ScriptComponent::script_handle);

            m_mesh_light_component.def(py::init<>())
                .def("setEmissiveColor", &atcg::MeshLightComponent::setEmissiveColor, "color"_a)
                .def("getEmissiveTexture", &atcg::MeshLightComponent::getEmissiveTexture)
                .def_readwrite("intensity", &atcg::MeshLightComponent::intensity)
                .def_readwrite("emissive_handle", &atcg::MeshLightComponent::emissive_handle);

            m_transparency.def(py::init<bool>(), "transparent"_a = true)
                .def_readwrite("transparent", &atcg::TransparencyComponent::transparent);

            m_homogeneous_medium.def(py::init<>())
                .def_readwrite("albedo", &atcg::HomogeneousMediumComponent::albedo)
                .def_readwrite("density", &atcg::HomogeneousMediumComponent::density)
                .def_readwrite("g", &atcg::HomogeneousMediumComponent::g)
                .def_readwrite("Le", &atcg::HomogeneousMediumComponent::Le)
                .def_readwrite("Le_color", &atcg::HomogeneousMediumComponent::Le_color);

            m_grid_component.def(py::init<>())
                .def_readwrite("bbox", &atcg::HeterogeneousMediumComponent::GridComponent::bbox)
                .def_readwrite("handle", &atcg::HeterogeneousMediumComponent::GridComponent::handle)
                .def_readwrite("scale", &atcg::HeterogeneousMediumComponent::GridComponent::scale);

            m_heterogeneous_medium.def(py::init<>())
                .def("density", &atcg::HeterogeneousMediumComponent::density)
                .def("albedo", &atcg::HeterogeneousMediumComponent::albedo)
                .def("emission", &atcg::HeterogeneousMediumComponent::emission)
                .def_readwrite("density_grid", &atcg::HeterogeneousMediumComponent::density_grid)
                .def_readwrite("albedo_grid", &atcg::HeterogeneousMediumComponent::albedo_grid)
                .def_readwrite("emission_grid", &atcg::HeterogeneousMediumComponent::emission_grid)
                .def_readwrite("g", &atcg::HeterogeneousMediumComponent::g);
        });
}
}    // namespace python
}    // namespace atcg
