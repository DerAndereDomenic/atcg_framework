#pragma once

#include <Core/Layer.h>

namespace atcg
{
/**
 * @brief This layer handles imgui context and is pushed as an overlay at the start of the application.
 * This is not meant to be used by the client
 */
class ImGuiLayer : public Layer
{
public:
    /**
     * @brief Construct a new Im Gui Layer object
     */
    ImGuiLayer();

    /**
     * @brief Destroy the Im Gui Layer object
     */
    ~ImGuiLayer();

    /**
     * @brief Initializes imgui -> only one ImGuiLayer should exist
     */
    virtual void onAttach() override;

    /**
     * @brief Destroy the layer
     */
    virtual void onDetach() override;

    /**
     * @brief Event callback
     *
     * @param event The event to handle
     */
    virtual void onEvent(Event* event) override;

    /**
     * @brief Start a new ImGui Frame
     */
    void begin();

    /**
     * @brief End the ImGui Frame
     */
    void end();

    /**
     * @brief Block events by imgui
     *
     * @param block If events should be blocked
     */
    inline void blockEvents(bool block) { _block_events = block; }

    /**
     * @brief Enable or disable Dock spaces
     *
     * @param enable If dockspaces should be enabled
     */
    inline void enableDockSpace(bool enable) { _enable_dock_space = enable; }

private:
    bool _block_events      = true;
    bool _enable_dock_space = false;
};
}    // namespace atcg