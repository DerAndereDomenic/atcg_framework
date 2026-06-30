#pragma once

#include <Core/API.h>
#include <Core/Memory.h>
#include <Core/SystemRegistry.h>
#include <Scene/Scene.h>
#include <Renderer/Camera.h>
#include <Renderer/GraphicsAPI.h>
#include <Plugin/PluginHandle.h>
#include <Renderer/Material.h>
#include <imgui.h>

#include <filesystem>
#include <unordered_map>
#include <functional>

namespace atcg
{

struct PluginInfo
{
    const char* name;
    const char* version;
    const char* author;
    const char* description;
};

class ATCG_API PluginManagerSystem
{
public:
    ~PluginManagerSystem();

    bool loadPlugin(const std::filesystem::path& path);

    bool releasePlugin(const std::filesystem::path& path);

private:
private:
    std::unordered_map<std::filesystem::path, PluginHandle> _loaded_plugins;
};

class ATCG_API PluginRegistry
{
public:
    PluginRegistry(PluginHandle handle) : _handle(handle) {}

    PluginRegistry(const PluginRegistry&)            = delete;
    PluginRegistry& operator=(const PluginRegistry&) = delete;

    template<typename MaterialT>
    void registerMaterial(std::string_view type)
    {
        MaterialRegistry::Registry* registry = MaterialRegistry::getRegistry();
        ATCG_REGISTER_MATERIAL_PLUGIN(registry, _handle, type, MaterialT);
    }

private:
    PluginHandle _handle;
};

#define ATCG_PLUGIN_LIBRARY()                                                                                          \
    extern "C" __declspec(dllexport) void registerSystems(ImGuiContext* imgui_context)                                 \
    {                                                                                                                  \
        ImGui::SetCurrentContext(imgui_context);                                                                       \
    }

}    // namespace atcg