#pragma once

#include <Events/Event.h>

namespace atcg
{
// Based on Hazel Engine (https://github.com/TheCherno/Hazel)
// Modified by Domenic Zingsheim in 2023

/**
 * @brief The layer system is used to handle events and draw calls in a specific order
 * In most of our cases we only have 1 layer -> the application layer that is created by the user.
 */
class Layer
{
public:
    /**
     * @brief Construct a new Layer object
     *
     * @param name The name of the layer
     */
    Layer(const std::string& name) : _debug_name(name) {}

    /**
     * @brief Destroy the Layer object
     *
     */
    virtual ~Layer() = default;

    /**
     * @brief This function gets called when the layer is created.
     * Implemented by user
     */
    virtual void onAttach() {}

    /**
     * @brief This function gets called when the layer is destroyed
     * Implemented by the user
     */
    virtual void onDetach() {}

    /**
     * @brief This function gets called every frame.
     * It is used for physics updates and rendering
     *
     * @param delta_time The time since the last frame
     */
    virtual void onUpdate(float delta_time) {}

    /**
     * @brief This function should only handle imgui code
     */
    virtual void onImGuiRender() {}

    /**
     * @brief The event handle callback
     *
     * @param event The event to handle
     */
    virtual void onEvent(Event* event) {}

    /**
     * @brief Get the Name object
     *
     * @return const std::string& The name of the layer
     */
    const std::string& getName() const { return _debug_name; }

protected:
    std::string _debug_name;
};
}    // namespace atcg