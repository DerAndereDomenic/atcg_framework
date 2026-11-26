#include "Bindings.h"

using namespace pybind11::literals;

namespace atcg
{
namespace python
{
// Timer, graphs, cameras, controllers, serialization and dictionaries
void bindDataStructure(BindingContext& ctx)
{
    py::module_ m = ctx.module();

    auto m_camera = py::class_<atcg::PerspectiveCamera, atcg::ref_ptr<atcg::PerspectiveCamera>>(m, "PerspectiveCamera");
    auto m_extrinsics = py::class_<atcg::CameraExtrinsics>(m, "CameraExtrinsics");
    auto m_intrinsics = py::class_<atcg::CameraIntrinsics>(m, "CameraIntrinsics");
    auto m_controller = py::class_<atcg::FirstPersonController, atcg::ref_ptr<atcg::FirstPersonController>>(m,
                                                                                                            "FirstPer"
                                                                                                            "sonContr"
                                                                                                            "oller");

    auto m_timer                = py::class_<atcg::Timer>(m, "Timer");
    auto m_vertex_specification = py::class_<atcg::VertexSpecification>(m, "VertexSpecification");
    auto m_edge_specification   = py::class_<atcg::EdgeSpecification>(m, "EdgeSpecification");
    auto m_graph                = py::class_<atcg::Graph, atcg::Asset, atcg::ref_ptr<atcg::Graph>>(m, "Graph");
    auto m_serializer           = py::class_<atcg::Serialization::SceneSerializer>(m, "SceneSerializer");

    auto m_dictionary = py::class_<atcg::Dictionary>(m, "Dictionary");

    ctx.define(
        [=]() mutable
        {
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
                .def_static(
                    "createTriangleMesh",
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
            m.def(
                "read_mesh",
                [](const std::string& path) { return atcg::IO::read_mesh(path); },
                "path"_a);

            m.def(
                "read_pointcloud",
                [](const std::string& path) { return atcg::IO::read_pointcloud(path); },
                "path"_a);

            m.def(
                "read_lines",
                [](const std::string& path) { return atcg::IO::read_lines(path); },
                "path"_a);

            m.def(
                "read_scene",
                [](const std::string& path) { return atcg::IO::read_scene(path); },
                "path"_a);

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
                .def(py::init<atcg::CameraExtrinsics, atcg::CameraIntrinsics>(),
                     "camera_extrinsics"_a,
                     "camera_intrinsics"_a)
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
                .def("serialize", &atcg::Serialization::SceneSerializer::serialize, "file_path"_a)
                .def("deserialize", &atcg::Serialization::SceneSerializer::deserialize, "file_path"_a);

            m_dictionary.def(py::init<>())
                .def("setInt8", &atcg::Dictionary::setInt8, "key"_a, "value"_a)
                .def("getInt8", &atcg::Dictionary::getInt8, "key"_a)
                .def("setInt16", &atcg::Dictionary::setInt16, "key"_a, "value"_a)
                .def("getInt16", &atcg::Dictionary::getInt16, "key"_a)
                .def("setInt32", &atcg::Dictionary::setInt32, "key"_a, "value"_a)
                .def("getInt32", &atcg::Dictionary::getInt32, "key"_a)
                .def("setInt64", &atcg::Dictionary::setInt64, "key"_a, "value"_a)
                .def("getInt64", &atcg::Dictionary::getInt64, "key"_a)
                .def("setUInt8", &atcg::Dictionary::setUInt8, "key"_a, "value"_a)
                .def("getUInt8", &atcg::Dictionary::getUInt8, "key"_a)
                .def("setUInt16", &atcg::Dictionary::setUInt16, "key"_a, "value"_a)
                .def("getUInt16", &atcg::Dictionary::getUInt16, "key"_a)
                .def("setUInt32", &atcg::Dictionary::setUInt32, "key"_a, "value"_a)
                .def("getUInt32", &atcg::Dictionary::getUInt32, "key"_a)
                .def("setUInt64", &atcg::Dictionary::setUInt64, "key"_a, "value"_a)
                .def("getUInt64", &atcg::Dictionary::getUInt64, "key"_a)
                .def("setFloat", &atcg::Dictionary::setFloat, "key"_a, "value"_a)
                .def("getFloat", &atcg::Dictionary::getFloat, "key"_a)
                .def("setDouble", &atcg::Dictionary::setDouble, "key"_a, "value"_a)
                .def("getDouble", &atcg::Dictionary::getDouble, "key"_a)
                .def("setString", &atcg::Dictionary::setString, "key"_a, "value"_a)
                .def("getString", &atcg::Dictionary::getString, "key"_a)
                .def("setTensor", &atcg::Dictionary::setTensor, "key"_a, "value"_a)
                .def("getTensor", &atcg::Dictionary::getTensor, "key"_a);
        });
}
}    // namespace python
}    // namespace atcg
