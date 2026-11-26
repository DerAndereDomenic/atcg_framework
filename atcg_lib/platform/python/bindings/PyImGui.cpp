#include "Bindings.h"

#ifndef ATCG_HEADLESS

using namespace pybind11::literals;

namespace atcg
{
namespace python
{
// ImGui, ImPlot and Guizmo
void bindImGui(BindingContext& ctx)
{
    py::module_ m = ctx.module();

    auto m_imgui            = m.def_submodule("ImGui");
    auto m_guizmo_operation = py::enum_<atcg::GuizmoOperation>(m_imgui, "GuizmoOperation");
    auto m_guizmo_mode      = py::enum_<atcg::GuizmoMode>(m_imgui, "GuizmoMode");

    // The layer is owned by the application, so python never deletes it. Layer uses the same holder.
    auto m_imgui_layer =
        py::class_<atcg::ImGuiLayer, atcg::Layer, std::unique_ptr<atcg::ImGuiLayer, py::nodelete>>(m, "ImGuiLayer");

    ctx.define(
        [=]() mutable
        {
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
                [](const char* label,
                   int* v,
                   int v_min,
                   int v_max,
                   const char* format     = "%d",
                   ImGuiSliderFlags flags = 0)
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
                    // The text is not used as a format string
                    ImGui::TextUnformatted(fmt);
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
                [](std::vector<std::vector<float>>& data, const std::vector<std::string>& names, const float line_width)
                {
                    if(names.size() < data.size())
                    {
                        ATCG_WARN("ImGui.plot: {0} names given for {1} lines", names.size(), data.size());
                    }

                    if(ImPlot::BeginPlot("##", ImVec2(-1, -1)))
                    {
                        ImPlot::SetupAxes("X", "Y", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                        ImPlot::SetupLegend(ImPlotLocation_North,
                                            ImPlotLegendFlags_Horizontal | ImPlotLegendFlags_Outside);
                        for(size_t i = 0; i < data.size(); ++i)
                        {
                            std::vector<float> x(data[i].size());
                            std::iota(x.begin(), x.end(), 0.0f);

                            ImPlot::SetNextLineStyle(IMPLOT_AUTO_COL, line_width);
                            std::string label = i < names.size() ? names[i] : "Line " + std::to_string(i);
                            ImPlot::PlotLine(label.c_str(),
                                             x.data(),
                                             data[i].data(),
                                             data[i].size(),
                                             0,
                                             0,
                                             sizeof(float));
                        }
                        ImPlot::EndPlot();
                    }
                },
                "data"_a,
                "names"_a,
                "line_width"_a = -1.0f);

            m_imgui.def("isUsing", &atcg::isUsingGuizmo);
            m_imgui.def("isOver", &atcg::isOverGuizmo);

            m_guizmo_operation.value("TRANSLATE", atcg::GuizmoOperation::TRANSLATE)
                .value("ROTATE", atcg::GuizmoOperation::ROTATE)
                .value("SCALE", atcg::GuizmoOperation::SCALE)
                .export_values();
            m_guizmo_mode.value("LOCAL", atcg::GuizmoMode::LOCAL)
                .value("WORLD", atcg::GuizmoMode::WORLD)
                .export_values();
            m_imgui.def("drawGuizmo", &atcg::drawGuizmo, "scene"_a, "operation"_a, "mode"_a, "camera"_a);

            m_imgui_layer.def("blockEvents", &atcg::ImGuiLayer::blockEvents, "block"_a)
                .def("enableDockSpace", &atcg::ImGuiLayer::enableDockSpace, "enable"_a)
                .def("dockspaceEnabled", &atcg::ImGuiLayer::dockspaceEnabled)
                .def("getViewportSize", &atcg::ImGuiLayer::getViewportSize)
                .def("getViewportPosition", &atcg::ImGuiLayer::getViewportPosition);
        });
}
}    // namespace python
}    // namespace atcg

#endif
