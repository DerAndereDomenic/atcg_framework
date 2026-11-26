#include "Bindings.h"

using namespace pybind11::literals;

namespace atcg
{
namespace python
{
// Editor panels
void bindGUI(BindingContext& ctx)
{
    py::module_ m = ctx.module();

    auto m_scene_hierarchy_panel = py::class_<atcg::GUI::SceneHierarchyPanel>(m, "SceneHierarchyPanel");

    auto m_performance_panel     = py::class_<atcg::GUI::PerformancePanel>(m, "PerformancePanel");
    auto m_render_settings_panel = py::class_<atcg::GUI::RenderSettingsPanel>(m, "RenderSettingsPanel");

    auto m_asset_panel = py::class_<atcg::GUI::AssetPanel, atcg::ref_ptr<atcg::GUI::AssetPanel>>(m, "AssetPanel");
#ifdef ATCG_CUDA_BACKEND
    auto m_integrator_panel = py::class_<atcg::GUI::IntegratorPanel>(m, "IntegratorPanel");
#endif

    ctx.define(
        [=]() mutable
        {
            m_asset_panel.def(py::init<>())
                .def("renderPanel", &atcg::GUI::AssetPanel::renderPanel)
                .def("selectAsset", &atcg::GUI::AssetPanel::selectAsset)
                .def("getSelectedAsset", &atcg::GUI::AssetPanel::getSelectedAsset)
                .def("isAssetSelectionActive", &atcg::GUI::AssetPanel::isAssetSelectionActive);

            m_performance_panel.def(py::init<>())
                .def(
                    "renderPanel",
                    [](atcg::GUI::PerformancePanel& panel, bool show_window)
                    {
                        panel.renderPanel(show_window);
                        return show_window;
                    },
                    "show_window"_a)
                .def("registerFrameTime", &atcg::GUI::PerformancePanel::registerFrameTime);

            m_render_settings_panel.def(py::init<>())
                .def(
                    "renderPanel",
                    [](atcg::GUI::RenderSettingsPanel& panel, bool show_window)
                    {
                        panel.renderPanel(show_window);
                        return show_window;
                    },
                    "show_window"_a);

            m_scene_hierarchy_panel.def(py::init<>())
                .def(py::init<>())
                .def("renderPanel", &atcg::GUI::SceneHierarchyPanel::renderPanel)
                .def("setEntitySelectionActive", &atcg::GUI::SceneHierarchyPanel::setEntitySelectionActive, "active"_a);

#ifdef ATCG_CUDA_BACKEND
            m_integrator_panel.def(py::init<>())
                .def(py::init<const atcg::ref_ptr<atcg::IntegratorController>&>(), "controller"_a)
                .def(
                    "renderPanel",
                    [](atcg::GUI::IntegratorPanel& panel, bool show_window)
                    {
                        panel.renderPanel(show_window);
                        return show_window;
                    },
                    "show_window"_a);
#endif
        });
}
}    // namespace python
}    // namespace atcg
