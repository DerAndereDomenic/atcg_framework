#include "Bindings.h"

using namespace pybind11::literals;

namespace atcg
{
namespace python
{
namespace
{
/**
 * @brief Bind a typed setter and getter for reference counted pointers stored in a dictionary.
 * The getter logs a warning and returns None if the key is missing or holds a different type.
 *
 * @tparam T The pointee type
 * @param dictionary The dictionary class
 * @param name The python suffix of the setter and getter
 */
template<typename T>
void bindDictionaryPointer(py::class_<atcg::Dictionary>& dictionary, const std::string& name)
{
    dictionary
        .def(("set" + name).c_str(),
             [](atcg::Dictionary& self, const std::string& key, const atcg::ref_ptr<T>& value)
             { self.setPointer<T>(key, value); },
             "key"_a,
             "value"_a)
        .def(("get" + name).c_str(),
             [](const atcg::Dictionary& self, const std::string& key) -> atcg::ref_ptr<T>
             {
                 if(!self.contains(key))
                 {
                     ATCG_WARN("Dictionary: key '{0}' not found", key);
                     return nullptr;
                 }
                 return self.getValueOr<atcg::ref_ptr<T>>(key, nullptr);
             },
             "key"_a);
}
}    // namespace

// Timer, graphs, bounding boxes, cameras, controllers, serialization and dictionaries
void bindDataStructure(BindingContext& ctx)
{
    py::module_ m = ctx.module();

    auto m_extrinsics = py::class_<atcg::CameraExtrinsics>(m, "CameraExtrinsics");
    auto m_intrinsics = py::class_<atcg::CameraIntrinsics>(m, "CameraIntrinsics");
    auto m_camera     = py::class_<atcg::Camera, atcg::ref_ptr<atcg::Camera>>(m, "Camera");
    auto m_perspective_camera =
        py::class_<atcg::PerspectiveCamera, atcg::Camera, atcg::ref_ptr<atcg::PerspectiveCamera>>(m,
                                                                                                  "PerspectiveCamera");
    auto m_orthographic_camera =
        py::class_<atcg::OrthographicCamera, atcg::Camera, atcg::ref_ptr<atcg::OrthographicCamera>>(m,
                                                                                                    "OrthographicCamer"
                                                                                                    "a");
    auto m_camera_controller =
        py::class_<atcg::CameraController, atcg::ref_ptr<atcg::CameraController>>(m, "CameraController");
    auto m_first_person_controller =
        py::class_<atcg::FirstPersonController, atcg::CameraController, atcg::ref_ptr<atcg::FirstPersonController>>(
            m,
            "FirstPersonController");
    auto m_focused_controller =
        py::class_<atcg::FocusedController, atcg::CameraController, atcg::ref_ptr<atcg::FocusedController>>(m,
                                                                                                            "FocusedCon"
                                                                                                            "troller");
    auto m_camera_utils = m.def_submodule("CameraUtils");

    auto m_timer                = py::class_<atcg::Timer>(m, "Timer");
    auto m_vertex_specification = py::class_<atcg::VertexSpecification>(m, "VertexSpecification");
    auto m_edge_specification   = py::class_<atcg::EdgeSpecification>(m, "EdgeSpecification");
    auto m_graph_type           = py::enum_<atcg::GraphType>(m, "GraphType");
    auto m_bounding_box         = py::class_<atcg::BoundingBox>(m, "BoundingBox");
    auto m_graph                = py::class_<atcg::Graph, atcg::Asset, atcg::ref_ptr<atcg::Graph>>(m, "Graph");
    auto m_serializer           = py::class_<atcg::Serialization::SceneSerializer>(m, "SceneSerializer");

    auto m_dictionary = py::class_<atcg::Dictionary>(m, "Dictionary");

#ifdef ATCG_CUDA_BACKEND
    auto m_jpeg_backend = py::enum_<atcg::JPEGBackend>(m, "JPEGBackend");
    auto m_jpeg_encoder = py::class_<atcg::JPEGEncoder, atcg::ref_ptr<atcg::JPEGEncoder>>(m, "JPEGEncoder");
    auto m_jpeg_decoder = py::class_<atcg::JPEGDecoder, atcg::ref_ptr<atcg::JPEGDecoder>>(m, "JPEGDecoder");
#endif

    ctx.define(
        [=]() mutable
        {
            m_timer.def(py::init<>())
                .def("elapsedMillis", &atcg::Timer::elapsedMillis)
                .def("elapsedSeconds", &atcg::Timer::elapsedSeconds)
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

            m_graph_type.value("None", atcg::GraphType::None)
                .value("ATCG_GRAPH_TYPE_POINTCLOUD", atcg::GraphType::ATCG_GRAPH_TYPE_POINTCLOUD)
                .value("ATCG_GRAPH_TYPE_TRIANGLEMESH", atcg::GraphType::ATCG_GRAPH_TYPE_TRIANGLEMESH)
                .value("ATCG_GRAPH_TYPE_GRAPH", atcg::GraphType::ATCG_GRAPH_TYPE_GRAPH);

            m_bounding_box.def(py::init<>())
                .def(py::init<>(
                         [](const glm::vec3& min, const glm::vec3& max)
                         {
                             atcg::BoundingBox bbox;
                             bbox.min = min;
                             bbox.max = max;
                             return bbox;
                         }),
                     "min"_a,
                     "max"_a)
                .def_readwrite("min", &atcg::BoundingBox::min)
                .def_readwrite("max", &atcg::BoundingBox::max)
                .def(
                    "__add__",
                    [](const atcg::BoundingBox& self, const atcg::BoundingBox& other) { return self + other; },
                    py::is_operator())
                .def("__repr__",
                     [](const atcg::BoundingBox& self)
                     {
                         return "BoundingBox(min=(" + std::to_string(self.min.x) + ", " + std::to_string(self.min.y) +
                                ", " + std::to_string(self.min.z) + "), max=(" + std::to_string(self.max.x) + ", " +
                                std::to_string(self.max.y) + ", " + std::to_string(self.max.z) + "))";
                     });

            m_graph.def(py::init<>())
                .def_static("createPointCloud", py::overload_cast<>(&atcg::Graph::createPointCloud))
                .def_static("createPointCloud",
                            py::overload_cast<const torch::Tensor&>(&atcg::Graph::createPointCloud),
                            "vertices"_a)
                .def_static("createTriangleMesh", py::overload_cast<>(&atcg::Graph::createTriangleMesh))
                .def_static(
                    "createTriangleMesh",
                    py::overload_cast<const torch::Tensor&, const torch::Tensor&>(&atcg::Graph::createTriangleMesh),
                    "vertices"_a,
                    "indices"_a)
                .def_static("createGraph", py::overload_cast<>(&atcg::Graph::createGraph))
                .def_static("createGraph",
                            py::overload_cast<const torch::Tensor&, const torch::Tensor&>(&atcg::Graph::createGraph),
                            "vertices"_a,
                            "edges"_a)
                .def_static("getStaticType", &atcg::Graph::getStaticType)
                .def("updateVertices",
                     py::overload_cast<const torch::Tensor&>(&atcg::Graph::updateVertices),
                     "vertices"_a)
                .def("updateFaces", py::overload_cast<const torch::Tensor&>(&atcg::Graph::updateFaces), "faces"_a)
                .def("updateEdges", py::overload_cast<const torch::Tensor&>(&atcg::Graph::updateEdges), "edges"_a)
                .def("resizeVertices", &atcg::Graph::resizeVertices, "size"_a)
                .def("resizeFaces", &atcg::Graph::resizeFaces, "size"_a)
                .def("resizeEdges", &atcg::Graph::resizeEdges, "size"_a)
                .def("getVerticesBuffer", &atcg::Graph::getVerticesBuffer)
                .def("getFaceIndexBuffer", &atcg::Graph::getFaceIndexBuffer)
                .def("getEdgesBuffer", &atcg::Graph::getEdgesBuffer)
                .def("getVerticesArray", &atcg::Graph::getVerticesArray)
                .def("getEdgesArray", &atcg::Graph::getEdgesArray)
                .def("getPositions", &atcg::Graph::getPositions, "device"_a)
                .def("getHostPositions", &atcg::Graph::getHostPositions)
                .def("getDevicePositions", &atcg::Graph::getDevicePositions)
                .def("getColors", &atcg::Graph::getColors, "device"_a)
                .def("getHostColors", &atcg::Graph::getHostColors)
                .def("getDeviceColors", &atcg::Graph::getDeviceColors)
                .def("getNormals", &atcg::Graph::getNormals, "device"_a)
                .def("getHostNormals", &atcg::Graph::getHostNormals)
                .def("getDeviceNormals", &atcg::Graph::getDeviceNormals)
                .def("getTangents", &atcg::Graph::getTangents, "device"_a)
                .def("getHostTangents", &atcg::Graph::getHostTangents)
                .def("getDeviceTangents", &atcg::Graph::getDeviceTangents)
                .def("getUVs", &atcg::Graph::getUVs, "device"_a)
                .def("getHostUVs", &atcg::Graph::getHostUVs)
                .def("getDeviceUVs", &atcg::Graph::getDeviceUVs)
                .def("getEdges", &atcg::Graph::getEdges, "device"_a)
                .def("getHostEdges", &atcg::Graph::getHostEdges)
                .def("getDeviceEdges", &atcg::Graph::getDeviceEdges)
                .def("getFaces", &atcg::Graph::getFaces, "device"_a)
                .def("getHostFaces", &atcg::Graph::getHostFaces)
                .def("getDeviceFaces", &atcg::Graph::getDeviceFaces)
                .def("n_vertices", &atcg::Graph::n_vertices)
                .def("n_faces", &atcg::Graph::n_faces)
                .def("n_edges", &atcg::Graph::n_edges)
                .def("type", &atcg::Graph::type)
                .def("getBoundingBox", &atcg::Graph::getBoundingBox)
                .def("updateBoundingBox", &atcg::Graph::updateBoundingBox)
                .def("copy", py::overload_cast<>(&atcg::Graph::copy, py::const_))
                .def("copy", py::overload_cast<const atcg::ref_ptr<atcg::Graph>&>(&atcg::Graph::copy), "other"_a)
                .def("unmapVertexPointer", &atcg::Graph::unmapVertexPointer)
                .def("unmapHostVertexPointer", &atcg::Graph::unmapHostVertexPointer)
                .def("unmapDeviceVertexPointer", &atcg::Graph::unmapDeviceVertexPointer)
                .def("unmapEdgePointer", &atcg::Graph::unmapEdgePointer)
                .def("unmapHostEdgePointer", &atcg::Graph::unmapHostEdgePointer)
                .def("unmapDeviceEdgePointer", &atcg::Graph::unmapDeviceEdgePointer)
                .def("unmapFacePointer", &atcg::Graph::unmapFacePointer)
                .def("unmapHostFacePointer", &atcg::Graph::unmapHostFacePointer)
                .def("unmapDeviceFacePointer", &atcg::Graph::unmapDeviceFacePointer)
                .def("unmapAllHostPointers", &atcg::Graph::unmapAllHostPointers)
                .def("unmapAllDevicePointers", &atcg::Graph::unmapAllDevicePointers)
                .def("unmapAllPointers", &atcg::Graph::unmapAllPointers);

            m.def(
                "read_any",
                [](const std::string& path) { return atcg::IO::read_any(path); },
                "path"_a);

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
                .def(py::init<const float, const float, const float, const float, const glm::vec2&>(),
                     "aspect_ratio"_a,
                     "fov_y"_a,
                     "znear"_a          = 0.01f,
                     "zfar"_a           = 1000.0f,
                     "optical_center"_a = glm::vec2(0))
                .def(py::init<const glm::mat4&>(), "projection"_a)
                .def("setAspectRatio", &atcg::CameraIntrinsics::setAspectRatio, "aspect_ratio"_a)
                .def("setFOV", &atcg::CameraIntrinsics::setFOV, "fov"_a)
                .def("setNear", &atcg::CameraIntrinsics::setNear, "near"_a)
                .def("setFar", &atcg::CameraIntrinsics::setFar, "far"_a)
                .def("setOpticalCenter", &atcg::CameraIntrinsics::setOpticalCenter, "optical_center"_a)
                .def("setProjection", &atcg::CameraIntrinsics::setProjection, "projection"_a)
                .def("setExposure", &atcg::CameraIntrinsics::setExposure, "exposure"_a)
                .def("aspectRatio", &atcg::CameraIntrinsics::aspectRatio)
                .def("FOV", &atcg::CameraIntrinsics::FOV)
                .def("zNear", &atcg::CameraIntrinsics::zNear)
                .def("zFar", &atcg::CameraIntrinsics::zFar)
                .def("opticalCenter", &atcg::CameraIntrinsics::opticalCenter)
                .def("projection", &atcg::CameraIntrinsics::projection)
                .def("getExposure", &atcg::CameraIntrinsics::getExposure);

            m_extrinsics.def(py::init<>())
                .def(py::init<const glm::vec3&, const glm::vec3&>(), "position"_a, "target"_a)
                .def(py::init<const glm::mat4&>(), "extrinsics_matrix"_a)
                .def("setPosition", &atcg::CameraExtrinsics::setPosition, "position"_a)
                .def("setTarget", &atcg::CameraExtrinsics::setTarget, "target"_a)
                .def("setExtrinsicMatrix", &atcg::CameraExtrinsics::setExtrinsicMatrix, "view"_a)
                .def("position", &atcg::CameraExtrinsics::position)
                .def("target", &atcg::CameraExtrinsics::target)
                .def("extrinsicMatrix", &atcg::CameraExtrinsics::extrinsicMatrix);

            m_camera.def("getProjection", &atcg::Camera::getProjection)
                .def("getViewProjection", &atcg::Camera::getViewProjection)
                .def("getView", &atcg::Camera::getView)
                .def("getPosition", &atcg::Camera::getPosition)
                .def("getDirection", &atcg::Camera::getDirection)
                .def("copy", &atcg::Camera::copy)
                .def("transformToCameraSpace", &atcg::Camera::transformToCameraSpace, "point"_a)
                .def("transformToNormalizedDeviceCoordinates",
                     &atcg::Camera::transformToNormalizedDeviceCoordinates,
                     "point"_a)
                .def("isPointInFrustum", &atcg::Camera::isPointInFrustum, "point"_a)
                .def("getExtrinsics", &atcg::Camera::getExtrinsics)
                .def("setExtrinsics", &atcg::Camera::setExtrinsics, "extrinsics"_a)
                .def("getIntrinsics", &atcg::Camera::getIntrinsics)
                .def("setIntrinsics", &atcg::Camera::setIntrinsics, "intrinsics"_a);

            m_perspective_camera
                .def(py::init<const atcg::CameraExtrinsics&, const atcg::CameraIntrinsics&>(),
                     "camera_extrinsics"_a = atcg::CameraExtrinsics(),
                     "camera_intrinsics"_a = atcg::CameraIntrinsics())
                .def("setPosition", &atcg::PerspectiveCamera::setPosition, "position"_a)
                .def("setView", &atcg::PerspectiveCamera::setView, "view"_a)
                .def("setProjection", &atcg::PerspectiveCamera::setProjection, "projection"_a)
                .def("getLookAt", &atcg::PerspectiveCamera::getLookAt)
                .def("setLookAt", &atcg::PerspectiveCamera::setLookAt, "look_at"_a)
                .def("getUp", &atcg::PerspectiveCamera::getUp)
                .def("getAspectRatio", &atcg::PerspectiveCamera::getAspectRatio)
                .def("setAspectRatio", &atcg::PerspectiveCamera::setAspectRatio, "aspect_ratio"_a)
                .def("setFOV", &atcg::PerspectiveCamera::setFOV, "fov"_a)
                .def("getFOV", &atcg::PerspectiveCamera::getFOV)
                .def("getNear", &atcg::PerspectiveCamera::getNear)
                .def("setNear", &atcg::PerspectiveCamera::setNear, "near_plane"_a)
                .def("getFar", &atcg::PerspectiveCamera::getFar)
                .def("setFar", &atcg::PerspectiveCamera::setFar, "far_plane"_a);

            m_orthographic_camera
                .def(py::init<const float&, const float&, const float&, const float&>(),
                     "left"_a,
                     "right"_a,
                     "bottom"_a,
                     "top"_a)
                .def("setProjection",
                     &atcg::OrthographicCamera::setProjection,
                     "left"_a,
                     "right"_a,
                     "bottom"_a,
                     "top"_a);

            m_camera_utils
                .def("convert_to_opencv",
                     py::overload_cast<const glm::mat4&, const uint32_t, const uint32_t>(
                         &atcg::CameraUtils::convert_to_opencv),
                     "projection"_a,
                     "width"_a,
                     "height"_a)
                .def("convert_to_opencv",
                     py::overload_cast<const atcg::CameraIntrinsics&, const uint32_t, const uint32_t>(
                         &atcg::CameraUtils::convert_to_opencv),
                     "intrinsics"_a,
                     "width"_a,
                     "height"_a)
                .def("convert_from_opencv",
                     py::overload_cast<const float,
                                       const float,
                                       const float,
                                       const float,
                                       const float,
                                       const float,
                                       const uint32_t,
                                       const uint32_t>(&atcg::CameraUtils::convert_from_opencv),
                     "fx"_a,
                     "fy"_a,
                     "cx"_a,
                     "cy"_a,
                     "n"_a,
                     "f"_a,
                     "width"_a,
                     "height"_a)
                .def("convert_from_opencv",
                     py::overload_cast<const glm::mat3&, const float, const float, const uint32_t, const uint32_t>(
                         &atcg::CameraUtils::convert_from_opencv),
                     "K"_a,
                     "n"_a,
                     "f"_a,
                     "width"_a,
                     "height"_a);

            m_camera_controller.def("onUpdate", &atcg::CameraController::onUpdate, "delta_time"_a)
                .def("onEvent", &atcg::CameraController::onEvent, "event"_a)
                .def("getCamera", &atcg::CameraController::getCamera)
                .def("setCamera", &atcg::CameraController::setCamera, "camera"_a);

            m_first_person_controller.def(py::init<const atcg::ref_ptr<atcg::PerspectiveCamera>&>(), "camera"_a);

            m_focused_controller.def(py::init<const atcg::ref_ptr<atcg::PerspectiveCamera>&>(), "camera"_a);

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
                .def("getTensor", &atcg::Dictionary::getTensor, "key"_a)
                .def("remove", &atcg::Dictionary::remove, "key"_a)
                .def("contains", &atcg::Dictionary::contains, "key"_a)
                .def("__contains__", &atcg::Dictionary::contains, "key"_a);

            bindDictionaryPointer<atcg::Scene>(m_dictionary, "Scene");
            bindDictionaryPointer<atcg::Graph>(m_dictionary, "Graph");
            bindDictionaryPointer<atcg::Texture2D>(m_dictionary, "Texture2D");
            bindDictionaryPointer<atcg::Texture3D>(m_dictionary, "Texture3D");
            bindDictionaryPointer<atcg::Material>(m_dictionary, "Material");

#ifdef ATCG_CUDA_BACKEND
            m_jpeg_backend.value("SOFTWARE", atcg::JPEGBackend::SOFTWARE)
                .value("HARDWARE", atcg::JPEGBackend::HARDWARE);

            m_jpeg_encoder
                .def(py::init<bool, atcg::JPEGBackend>(),
                     "flip_vertically"_a = true,
                     "backend"_a         = atcg::JPEGBackend::SOFTWARE)
                .def("compress", &atcg::JPEGEncoder::compress, "img"_a);

            m_jpeg_decoder
                .def(py::init<uint32_t, uint32_t, uint32_t, bool, atcg::JPEGBackend>(),
                     "num_images"_a,
                     "img_width"_a,
                     "img_height"_a,
                     "flip_vertically"_a = true,
                     "backend"_a         = atcg::JPEGBackend::SOFTWARE)
                .def(
                    "decompressImages",
                    [](const atcg::ref_ptr<atcg::JPEGDecoder>& self, const std::vector<py::bytes>& jpeg_files)
                    {
                        std::vector<std::vector<uint8_t>> files;
                        files.reserve(jpeg_files.size());
                        for(const auto& file: jpeg_files)
                        {
                            std::string data = file.cast<std::string>();
                            files.emplace_back(data.begin(), data.end());
                        }
                        return self->decompressImages(files);
                    },
                    "jpeg_files"_a)
                .def("copyToOutput",
                     py::overload_cast<const atcg::ref_ptr<atcg::Texture3D>&>(&atcg::JPEGDecoder::copyToOutput),
                     "texture"_a);
#endif
        });
}
}    // namespace python
}    // namespace atcg
