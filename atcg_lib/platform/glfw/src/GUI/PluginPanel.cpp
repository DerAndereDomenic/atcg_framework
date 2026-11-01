#include <GUI/PluginPanel.h>

#include <imgui.h>

#include <portable-file-dialogs.h>

#include <Plugin/Plugin.h>

namespace atcg
{
namespace GUI
{
void PluginPanel::renderPanel()
{
    const ImGuiTreeNodeFlags treeNodeFlags = ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth |
                                             ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_FramePadding;

    const auto& loaded_plugins = atcg::PluginManager::getLoadedPlugins();
    ImGui::Begin("Plugins");

    for(auto [path, handle]: loaded_plugins)
    {
        std::string plugin_name = path.filename().stem().string();
        bool open               = ImGui::TreeNodeEx(plugin_name.c_str(), treeNodeFlags);
        if(open)
        {
            ImGui::Text("Path: %s", path.string().c_str());
            ImGui::Text("Handle: %p", handle);

            if(ImGui::Button(("Reload##" + plugin_name).c_str()))
            {
                atcg::PluginManager::releasePlugin(path);
                atcg::PluginManager::loadPlugin(path);

                ImGui::TreePop();
                break;    // Break the loop to avoid invalidating the iterator after reloading the plugin
            }

            if(ImGui::Button(("Unload##" + plugin_name).c_str()))
            {
                atcg::PluginManager::releasePlugin(path);
                ImGui::TreePop();
                break;    // Break the loop to avoid invalidating the iterator after unloading the plugin
            }

            ImGui::TreePop();
        }
    }

    if(ImGui::Button("Load Plugin"))
    {
        auto f     = pfd::open_file("Choose plugin to load",
                                    pfd::path::home(),
                                    {"Dynamic Library (.dll, .so)", "*.dll *.so"},
                                    pfd::opt::none);
        auto files = f.result();
        if(!files.empty())
        {
            atcg::PluginManager::loadPlugin(files[0]);
        }
    }

    if(ImGui::Button("Unload All Plugins"))
    {
        atcg::PluginManager::releaseAllPlugins();
    }


    ImGui::End();
}
}    // namespace GUI
}    // namespace atcg