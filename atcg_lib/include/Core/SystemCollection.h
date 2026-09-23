#pragma once

#include <Core/Memory.h>
#include <Core/Window.h>

namespace atcg
{
/**
 * @brief A class that handles all the systems used by the application
 */
class SystemCollection
{
public:
    /**
     * @brief Constructor
     */
    SystemCollection();

    /**
     * @brief Destructor
     */
    ~SystemCollection();

    /**
     * @brief Initializes the core systems
     *
     * @param props The window properties
     * @param event_callback The event callback function
     */
    void initSystems(const WindowProps& props, const Window::EventCallbackFn& event_callback);

    /**
     * @brief Shuts down the core systems
     */
    void shutdownSystems();

    /**
     * @brief Get the window
     *
     * @return The window
     */
    const atcg::scope_ptr<Window>& getWindow() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg