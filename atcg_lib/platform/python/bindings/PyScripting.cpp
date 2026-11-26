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

    auto m_scriptengine_base = py::class_<atcg::ScriptEngine, atcg::ref_ptr<atcg::ScriptEngine>>(m, "ScriptEngineBase");
    using PythonScriptEngineClass =
        py::class_<atcg::PythonScriptEngine, atcg::ScriptEngine, atcg::ref_ptr<atcg::PythonScriptEngine>>;
    auto m_scriptengine = PythonScriptEngineClass(m, "ScriptEngine");
    auto m_script_base  = py::class_<atcg::Script, atcg::Asset, atcg::ref_ptr<atcg::Script>>(m, "ScriptBase");
    auto m_script       = py::class_<atcg::PythonScript, atcg::Script, atcg::ref_ptr<atcg::PythonScript>>(m, "Script");

    ctx.define(
        [=]() mutable
        {
            m_behavior.def(py::init<>())
                .def("onAttach", &atcg::Behavior::onAttach)
                .def("onDetach", &atcg::Behavior::onDetach)
                .def("onUpdate", &atcg::Behavior::onUpdate, "delta_time"_a)
                .def("onImGuiRender", &atcg::Behavior::onImGuiRender)
                .def("onEvent", &atcg::Behavior::onEvent, "event"_a);

            m_scriptengine_base.def("init", &atcg::ScriptEngine::init).def("destroy", &atcg::ScriptEngine::destroy);

            m_scriptengine.def(py::init<>());

            m_script_base.def("init", &atcg::Script::init)
                .def("createBehavior", &atcg::Script::createBehavior, "scene"_a, "entity"_a)
                .def("reload", &atcg::Script::reload)
                .def("getFilePath", [](const atcg::Script& script) { return script.getFilePath().string(); })
                .def("getSource", &atcg::Script::getSource)
                .def_static("getStaticType", &atcg::Script::getStaticType);

            m_script.def(
                py::init([](const std::string& file_path) { return atcg::make_ref<atcg::PythonScript>(file_path); }),
                "file_path"_a);

            m.def("handleScriptReloads", &atcg::Scripting::handleScriptReloads, "scene"_a);
            m.def("handleScriptEvents", &atcg::Scripting::handleScriptEvents, "scene"_a, "event"_a);
            m.def("handleScriptUpdates", &atcg::Scripting::handleScriptUpdates, "scene"_a, "dt"_a);
        });
}
}    // namespace python
}    // namespace atcg
