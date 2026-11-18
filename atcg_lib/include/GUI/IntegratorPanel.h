#pragma once

#ifdef ATCG_CUDA_BACKEND

    #include <Core/API.h>
    #include <Core/Memory.h>
    #include <Integrator/IntegratorController.h>

namespace atcg
{
namespace GUI
{
/**
 * @brief A panel to start and stop path tracing, select the integrator and edit its settings.
 * The panel holds no state. All commands are sent to the IntegratorController that does the actual work.
 */
class ATCG_API IntegratorPanel
{
public:
    /**
     * @brief Default constructor. The panel draws nothing until a controller is assigned.
     */
    IntegratorPanel() = default;

    /**
     * @brief Constructor
     *
     * @param controller The controller of the integrator
     */
    IntegratorPanel(const atcg::ref_ptr<IntegratorController>& controller) : _controller(controller) {}

    /**
     * @brief Render the panel
     *
     * @param show_window If the window should be shown. Set to false if the window is closed.
     */
    void renderPanel(bool& show_window);

private:
    atcg::ref_ptr<IntegratorController> _controller;
};
}    // namespace GUI
}    // namespace atcg

#endif
