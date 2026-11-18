#pragma once

#include <Core/API.h>

namespace atcg
{
namespace GUI
{
/**
 * @brief A panel to change the settings of the rasterization renderer (vsync, MSAA, shadow map resolution).
 * The panel holds no state. It reads and writes the settings of the window and the SceneRenderer directly.
 */
class ATCG_API RenderSettingsPanel
{
public:
    /**
     * @brief Default constructor
     */
    RenderSettingsPanel() = default;

    /**
     * @brief Render the panel
     *
     * @param show_window If the window should be shown. Set to false if the window is closed.
     */
    void renderPanel(bool& show_window);
};
}    // namespace GUI
}    // namespace atcg
