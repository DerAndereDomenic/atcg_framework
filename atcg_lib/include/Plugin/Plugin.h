#pragma once

#include <Core/API.h>
#include <Core/Memory.h>
#include <Core/SystemRegistry.h>
#include <Scene/Scene.h>
#include <Renderer/Camera.h>
#include <Renderer/GraphicsAPI.h>
#include <Plugin/PluginHandle.h>
#include <Renderer/Material.h>
#ifdef ATCG_CUDA_BACKEND
    #include <BSDF/BSDFRegistry.h>
#endif
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

    bool releaseAllPlugins();

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

#ifdef ATCG_CUDA_BACKEND
    template<typename BSDFT>
    void registerBSDF(std::string_view type)
    {
        BSDFRegistry::Registry* registry = BSDFRegistry::getRegistry();
        ATCG_REGISTER_BSDF_PLUGIN(registry, _handle, type, BSDFT);
    }
#endif

private:
    PluginHandle _handle;
};

namespace PluginManager
{
ATCG_INLINE bool loadPlugin(const std::filesystem::path& path)
{
    return SystemRegistry::instance()->getSystem<PluginManagerSystem>()->loadPlugin(path);
}

ATCG_INLINE bool releasePlugin(const std::filesystem::path& path)
{
    return SystemRegistry::instance()->getSystem<PluginManagerSystem>()->releasePlugin(path);
}

ATCG_INLINE bool releaseAllPlugins()
{
    return SystemRegistry::instance()->getSystem<PluginManagerSystem>()->releaseAllPlugins();
}
}    // namespace PluginManager

#define ATCG_PLUGIN_LIBRARY()                                                                                          \
    extern "C" __declspec(dllexport) void registerSystems(ImGuiContext* imgui_context)                                 \
    {                                                                                                                  \
        ImGui::SetCurrentContext(imgui_context);                                                                       \
    }

}    // namespace atcg