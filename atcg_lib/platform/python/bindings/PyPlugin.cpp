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
                .def(
                    "loadPlugin",
                    [](const std::string& path) { return atcg::PluginManager::loadPlugin(path); },
                    "path"_a)
                .def(
                    "releasePlugin",
                    [](const std::string& path) { return atcg::PluginManager::releasePlugin(path); },
                    "path"_a)
                .def("releaseAllPlugins", &atcg::PluginManager::releaseAllPlugins)
                .def(
                    "loadPluginFromName",
                    [](const std::string& name) { return atcg::PluginManager::loadPluginFromName(name); },
                    "name"_a)
                .def(
                    "releasePluginFromName",
                    [](const std::string& name) { return atcg::PluginManager::releasePluginFromName(name); },
                    "name"_a)
                .def("getLoadedPlugins",
                     []()
                     {
                         std::vector<std::string> paths;
                         for(const auto& [path, handle]: atcg::PluginManager::getLoadedPlugins())
                         {
                             paths.push_back(path.string());
                         }
                         return paths;
                     });
        });
}
}    // namespace python
}    // namespace atcg
