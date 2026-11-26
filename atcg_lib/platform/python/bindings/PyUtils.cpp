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
                     [](const atcg::ref_ptr<atcg::Scene>& scene,
                        const atcg::ref_ptr<atcg::PerspectiveCamera>& cam,
                        const uint32_t width) { return atcg::Utils::screenshot(scene, cam, width); })
                .def(
                    "screenshot",
                    [](const atcg::ref_ptr<atcg::Scene>& scene,
                       const atcg::ref_ptr<atcg::PerspectiveCamera>& cam,
                       const uint32_t width,
                       const std::string& path) { atcg::Utils::screenshot(scene, cam, width, path); },
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
                       const std::string& path) { atcg::Utils::screenshot(scene, cam, width, height, path); },
                    "scene"_a,
                    "camera"_a,
                    "width"_a,
                    "height"_a,
                    "path"_a)
                .def("pickEntity", &atcg::Utils::pickEntity, "mouse_pos"_a);

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
