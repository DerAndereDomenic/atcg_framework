#include <GUI/PerformancePanel.h>

namespace atcg
{
namespace GUI
{
void PerformancePanel::renderPanel(bool& show_window) {}

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