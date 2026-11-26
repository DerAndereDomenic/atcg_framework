#include "Bindings.h"

using namespace pybind11::literals;

namespace atcg
{
namespace python
{
// Utils submodule
void bindUtils(BindingContext& ctx)
{
    py::module_ m = ctx.module();

    auto m_utils = m.def_submodule("Utils");

    ctx.define(
        [=]() mutable
        {
            m_utils
                .def("screenshot",
                     py::overload_cast<const atcg::ref_ptr<atcg::Scene>&, const atcg::ref_ptr<atcg::Camera>&, uint32_t>(
                         &atcg::Utils::screenshot),
                     "scene"_a,
                     "camera"_a,
                     "width"_a)
                .def("screenshot",
                     py::overload_cast<const atcg::ref_ptr<atcg::Scene>&,
                                       const atcg::ref_ptr<atcg::Camera>&,
                                       uint32_t,
                                       const std::string&>(&atcg::Utils::screenshot),
                     "scene"_a,
                     "camera"_a,
                     "width"_a,
                     "path"_a)
                .def("screenshot",
                     py::overload_cast<const atcg::ref_ptr<atcg::Scene>&,
                                       const atcg::ref_ptr<atcg::Camera>&,
                                       uint32_t,
                                       uint32_t,
                                       const std::string&>(&atcg::Utils::screenshot),
                     "scene"_a,
                     "camera"_a,
                     "width"_a,
                     "height"_a,
                     "path"_a)
                .def("pickEntity", &atcg::Utils::pickEntity, "mouse_pos"_a);

            // Geometry
            m_utils.def("normalize",
                        py::overload_cast<const atcg::ref_ptr<atcg::Graph>&>(&atcg::Utils::normalize),
                        "graph"_a);
            m_utils.def("normalize",
                        py::overload_cast<const atcg::ref_ptr<atcg::Graph>&, atcg::TransformComponent&>(
                            &atcg::Utils::normalize),
                        "graph"_a,
                        "transform"_a);
            m_utils.def("applyTransform",
                        py::overload_cast<const atcg::ref_ptr<atcg::Graph>&, atcg::TransformComponent&>(
                            &atcg::Utils::applyTransform),
                        "graph"_a,
                        "transform"_a);
            m_utils.def(
                "applyTransform",
                [](torch::Tensor positions,
                   torch::Tensor normals,
                   torch::Tensor tangents,
                   atcg::TransformComponent& transform)
                { atcg::Utils::applyTransform(positions, normals, tangents, transform); },
                "positions"_a,
                "normals"_a,
                "tangents"_a,
                "transform"_a);

            // Bounding boxes
            m_utils.def("transformBoundingBox", &atcg::Utils::transformBoundingBox, "bbox"_a, "transform"_a);
            m_utils.def("boundingBoxToModelMatrix", &atcg::Utils::boundingBoxToModelMatrix, "bbox"_a);
            m_utils.def("insideBoundingBox", &atcg::Utils::insideBoundingBox, "point"_a, "bbox"_a);
            m_utils.def("getBoundingBoxCorners", &atcg::Utils::getBoundingBoxCorners, "bbox"_a);
            m_utils.def("isVisible", &atcg::Utils::isVisible, "camera"_a, "bbox"_a);

            // Binary IO
            m_utils.def("dumpBinary", &atcg::Utils::dumpBinary, "path"_a, "data"_a);
            m_utils.def(
                "serializeBuffer",
                [](const std::string& file_name, const py::bytes& data)
                {
                    std::string buffer = data;
                    atcg::Utils::serializeBuffer(file_name, buffer.data(), (uint32_t)buffer.size());
                },
                "file_name"_a,
                "data"_a);
            m_utils.def(
                "deserializeBuffer",
                [](const std::string& file_name)
                {
                    std::vector<uint8_t> buffer = atcg::Utils::deserializeBuffer(file_name);
                    return py::bytes(reinterpret_cast<const char*>(buffer.data()), buffer.size());
                },
                "file_name"_a);

            // Assets
            m_utils.def("canApplyAsset", &atcg::Utils::canApplyAsset, "entity"_a, "handle"_a);
            m_utils.def("applyAssetToEntities",
                        &atcg::Utils::applyAssetToEntities,
                        "scene"_a,
                        "entities"_a,
                        "handle"_a);
            m_utils.def("createEntityFromAsset", &atcg::Utils::createEntityFromAsset, "scene"_a, "handle"_a);
            m_utils.def("handleViewportAssetDrop",
                        &atcg::Utils::handleViewportAssetDrop,
                        "scene"_a,
                        "handle"_a,
                        "mouse_pos"_a);

#ifndef ATCG_HEADLESS
            // GUI helpers. The bool& out parameters are returned as tuples.
            m_utils.def(
                "displayMaterialSelection",
                [](const std::string& key, atcg::AssetHandle handle)
                {
                    bool deactivated = false;
                    auto result      = atcg::Utils::displayMaterialSelection(key, handle, deactivated);
                    return std::make_tuple(result, deactivated);
                },
                "key"_a,
                "handle"_a);
            m_utils.def(
                "displayGraphSelection",
                [](const std::string& key, atcg::AssetHandle handle)
                {
                    bool deactivated = false;
                    auto result      = atcg::Utils::displayGraphSelection(key, handle, deactivated);
                    return std::make_tuple(result, deactivated);
                },
                "key"_a,
                "handle"_a);
            m_utils.def(
                "displayScriptSelection",
                [](const std::string& key, atcg::AssetHandle handle)
                {
                    bool deactivated = false;
                    auto result      = atcg::Utils::displayScriptSelection(key, handle, deactivated);
                    return std::make_tuple(result, deactivated);
                },
                "key"_a,
                "handle"_a);
            m_utils.def(
                "displayShaderSelection",
                [](const std::string& key, atcg::AssetHandle handle)
                {
                    bool deactivated = false;
                    auto result      = atcg::Utils::displayShaderSelection(key, handle, deactivated);
                    return std::make_tuple(result, deactivated);
                },
                "key"_a,
                "handle"_a);
            m_utils.def(
                "displayTexture2DSelection",
                [](const std::string& key, atcg::AssetHandle handle)
                {
                    bool deactivated = false;
                    auto result      = atcg::Utils::displayTexture2DSelection(key, handle, deactivated);
                    return std::make_tuple(result, deactivated);
                },
                "key"_a,
                "handle"_a);
            m_utils.def(
                "displayTexture3DSelection",
                [](const std::string& key, atcg::AssetHandle handle)
                {
                    bool deactivated = false;
                    auto result      = atcg::Utils::displayTexture3DSelection(key, handle, deactivated);
                    return std::make_tuple(result, deactivated);
                },
                "key"_a,
                "handle"_a);
            m_utils.def(
                "displayCullModeSelection",
                [](const std::string& key, atcg::CullMode cull_mode)
                {
                    bool deactivated = false;
                    auto result      = atcg::Utils::displayCullModeSelection(key, cull_mode, deactivated);
                    return std::make_tuple(result, deactivated);
                },
                "key"_a,
                "cull_mode"_a);
            m_utils.def("beginAssetDragSource", &atcg::Utils::beginAssetDragSource, "handle"_a);
            m_utils.def(
                "acceptAssetDrop",
                [](atcg::AssetType type)
                {
                    atcg::AssetHandle handle(0);
                    bool accepted = atcg::Utils::acceptAssetDrop(type, handle);
                    return std::make_tuple(accepted, handle);
                },
                "type"_a);
            m_utils.def(
                "acceptAssetDrop",
                [](const std::function<bool(atcg::AssetHandle)>& filter)
                {
                    atcg::AssetHandle handle(0);
                    bool accepted = atcg::Utils::acceptAssetDrop(handle, filter);
                    return std::make_tuple(accepted, handle);
                },
                "filter"_a = py::none());
#endif

            // Image metrics
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
        });
}
}    // namespace python
}    // namespace atcg
