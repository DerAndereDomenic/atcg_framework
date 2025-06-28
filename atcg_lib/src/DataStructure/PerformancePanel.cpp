#include <DataStructure/PerformancePanel.h>

#ifndef ATCG_HEADLESS
    #include <implot.h>
#endif

namespace atcg
{
namespace GUI
{
void PerformancePanel::renderPanel(bool& show_window)
{
#ifndef ATCG_HEADLESS
    if(show_window)
    {
        ImGui::Begin("Performance##atcg_lib", &show_window);

        ImGui::Text(
            ("FPS: " + std::to_string((int)(1.0f / (_frame_time_collection_mean.mean() / 1000.0f))) + " fps").c_str());
        ImGui::Separator();

        ImPlot::SetNextAxisLimits(ImAxis_Y1, 0, 40);
        if(ImPlot::BeginPlot("Runtime", ImVec2(-1, 0), ImPlotFlags_NoLegend))
        {
            ImPlot::SetupAxes("X", "Y", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_None);
            ImPlot::PushStyleColor(ImPlotCol_Line, {0.0f, 1.0f, 0.0f, 0.5f});
            constexpr float val[2] = {16.666f, 33.3333f};
            ImPlot::PlotInfLines("Targets", val, 2, ImPlotInfLinesFlags_Horizontal);
            ImPlot::PushStyleColor(ImPlotCol_Line, {1.0f, 0.0f, 0.0f, 0.3f});
            ImPlot::PlotLine("Frame Time",
                             _frame_id_collection.get(),
                             _frame_time_collection.get(),
                             _frame_time_collection.count(),
                             0,
                             _frame_time_collection.index(),
                             sizeof(float));
            ImPlot::PushStyleColor(ImPlotCol_Line, {1.0f, 0.0f, 0.0f, 1.0f});
            ImPlot::PlotLine("Frame Time",
                             _frame_id_collection.get(),
                             _frame_time_collection_smooth.get(),
                             _frame_time_collection_smooth.count(),
                             0,
                             _frame_time_collection_smooth.index(),
                             sizeof(float));
            ImPlot::PopStyleColor(3);
            ImPlot::EndPlot();
        }

        ImGui::End();
    }
#endif
}

void PerformancePanel::registerFrameTime(const float frame_time)
{
    _frame_id_collection.addSample((float)_frame_id);
    _frame_time_collection_mean.addSample(frame_time * 1000.0f);
    _frame_time_collection_smooth.addSample(_frame_time_collection_mean.mean());
    _frame_time_collection.addSample(frame_time * 1000.0f);
    ++_frame_id;
}
}    // namespace GUI
}    // namespace atcg