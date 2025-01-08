#pragma once

#include <DataStructure/Statistics.h>

namespace atcg
{

/**
 * @brief A debug panel to display frame times
 */
class PerformancePanel
{
public:
    /**
     * @brief Default constructor
     */
    PerformancePanel() = default;

    /**
     * @brief Render the panel
     *
     * @param show_window If the window should be shown
     */
    void renderPanel(bool& show_window);

    /**
     * @brief Register a new frame time sample (in seconds)
     *
     * @param frame_time The new frame time in seconds
     */
    void registerFrameTime(const float frame_time);

private:
    atcg::CyclicCollection<float> _frame_id_collection = atcg::CyclicCollection<float>("Frame ID Collection", 1000);
    atcg::CyclicCollection<float> _frame_time_collection_smooth =
        atcg::CyclicCollection<float>("Frame Time Collection", 1000);
    atcg::CyclicCollection<float> _frame_time_collection = atcg::CyclicCollection<float>("Frame Time Collection", 1000);
    atcg::CyclicCollection<float> _frame_time_collection_mean =
        atcg::CyclicCollection<float>("Frame Time Collection", 5);
    uint32_t _frame_id = 0;
};
}    // namespace atcg