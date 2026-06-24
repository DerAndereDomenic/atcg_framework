#pragma once

#include <Core/API.h>
#include <Core/Memory.h>
#include <Core/Platform.h>
#include <Events/Event.h>
#include <Scene/Entity.h>
#include <Asset/Asset.h>
#include <pybind11/pybind11.h>

namespace atcg
{
/**
 * @brief A class that models the behavior of an entity
 */
class ATCG_API Behavior
{
public:
    /**
     * @brief Default constructor
     */
    Behavior() = default;

    /**
     * @brief Called when the behavior is attached
     */
    virtual void onAttach() {}

    /**
     * @brief Called when the behavior is detached
     */
    virtual void onDetach() {}

    /**
     * @brief Called each frame
     *
     * @param delta_time The delta time between frames
     */
    virtual void onUpdate(float delta_time) {}

    /**
     * @brief Called for imgui
     */
    virtual void onImGuiRender() {}

    /**
     * @brief Called on event handling
     *
     * @param event The event
     */
    virtual void onEvent(Event* event) {}

private:
};

/**
 * @brief Behavior incapsulated into a python script
 */
class ATCG_API PythonBehavior : public Behavior
{
public:
    /**
     * @brief Called when the behavior is attached
     */
    virtual void onAttach() override;

    /**
     * @brief Called when the behavior is detached
     */
    virtual void onDetach() override;

    /**
     * @brief Called each frame
     *
     * @param delta_time The delta time between frames
     */
    virtual void onUpdate(float delta_time) override;

    /**
     * @brief Called for imgui
     */
    virtual void onImGuiRender() override;

    /**
     * @brief Called on event handling
     *
     * @param event The event
     */
    virtual void onEvent(Event* event) override;

    /**
     * @brief Stores the python object that represents this class
     *
     * @param self The python instance of this class
     */
    ATCG_INLINE void setSelf(const pybind11::object& self) { _instance = self; }

private:
    pybind11::object _instance;
};
}    // namespace atcg