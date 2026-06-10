#include <Scripting/Behavior.h>

#include <pybind11/pybind11.h>

namespace atcg
{
void PythonBehavior::onAttach()
{
    PYBIND11_OVERRIDE(void, atcg::Behavior, onAttach);
}

void PythonBehavior::onDetach()
{
    PYBIND11_OVERRIDE(void, atcg::Behavior, onDetach);
}

void PythonBehavior::onUpdate(float delta_time)
{
    PYBIND11_OVERRIDE(void, atcg::Behavior, onUpdate, delta_time);
}

void PythonBehavior::onImGuiRender()
{
    PYBIND11_OVERRIDE(void, atcg::Behavior, onImGuiRender);
}

void PythonBehavior::onEvent(Event* event)
{
    PYBIND11_OVERRIDE(void, atcg::Behavior, onEvent, event);
}
}    // namespace atcg