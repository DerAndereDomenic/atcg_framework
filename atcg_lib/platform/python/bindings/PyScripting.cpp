#include "Bindings.h"

using namespace pybind11::literals;

namespace atcg
{
namespace python
{
// Behaviors, scripts and the script engine
void bindScripting(BindingContext& ctx)
{
    py::module_ m = ctx.module();

    auto m_behavior = py::class_<atcg::Behavior, atcg::PythonBehavior, std::shared_ptr<atcg::Behavior>>(m, "Behavior");

    auto m_scriptengine =
        py::class_<atcg::PythonScriptEngine, atcg::ref_ptr<atcg::PythonScriptEngine>>(m, "ScriptEngine");
    auto m_script = py::class_<atcg::PythonScript, atcg::ref_ptr<atcg::PythonScript>>(m, "Script");

    ctx.define(
        [=]() mutable
        {
            m_behavior.def(py::init<>())
                .def("onAttach", &atcg::Behavior::onAttach)
                .def("onUpdate", &atcg::Behavior::onUpdate, "delta_time"_a)
                .def("onImGuiRender", &atcg::Behavior::onImGuiRender)
                .def("onEvent", &atcg::Behavior::onEvent, "event"_a);

            m_scriptengine.def(py::init<>())
                .def("init", &atcg::PythonScriptEngine::init)
                .def("destroy", &atcg::PythonScriptEngine::destroy);

            m_script.def(py::init<const std::filesystem::path&>())
                .def("init", &atcg::PythonScript::init)
                .def("createBehavior", &atcg::PythonScript::createBehavior)
                .def("reload", &atcg::PythonScript::reload);

            m.def("handleScriptReloads", &atcg::Scripting::handleScriptReloads);
            m.def("handleScriptEvents", &atcg::Scripting::handleScriptEvents);
            m.def("handleScriptUpdates", &atcg::Scripting::handleScriptUpdates);
        });
}
}    // namespace python
}    // namespace atcg
