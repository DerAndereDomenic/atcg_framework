#include "Bindings.h"

using namespace pybind11::literals;

namespace atcg
{
namespace python
{
// Plugin manager
void bindPlugin(BindingContext& ctx)
{
    py::module_ m = ctx.module();

    auto m_plugin_manager = m.def_submodule("PluginManager");

    ctx.define(
        [=]() mutable
        {
            m_plugin_manager
                .def("loadPlugin", [](const std::string& path) { return atcg::PluginManager::loadPlugin(path); })
                .def("releasePlugin", [](const std::string& path) { return atcg::PluginManager::releasePlugin(path); })
                .def("releaseAllPlugins", &atcg::PluginManager::releaseAllPlugins)
                .def("loadPluginFromName",
                     [](const std::string& name) { return atcg::PluginManager::loadPluginFromName(name); })
                .def("releasePluginFromName",
                     [](const std::string& name) { return atcg::PluginManager::releasePluginFromName(name); });
        });
}
}    // namespace python
}    // namespace atcg
