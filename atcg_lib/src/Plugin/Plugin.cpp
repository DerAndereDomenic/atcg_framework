#include <Plugin/Plugin.h>
#include <Windows.h>    // TODO

namespace atcg
{

PluginManagerSystem::~PluginManagerSystem()
{
    releaseAllPlugins();
}

bool PluginManagerSystem::loadPlugin(const std::filesystem::path& path)
{
    // TODO Platform-specific implementation for loading plugin using LoadLibrary (Windows) or dlopen (Linux)
    // Implementation for loading plugin

    auto handle = LoadLibraryA(path.string().c_str());
    if(!handle)
    {
        // Handle error
        ATCG_ERROR("Failed to load plugin: {0}", path.string());
        return false;
    }

    using RegisterPluginFunc = void (*)(PluginRegistry&);
    auto registerPlugin =
        reinterpret_cast<RegisterPluginFunc>(GetProcAddress(static_cast<HMODULE>(handle), "registerPlugin"));
    if(!registerPlugin)
    {
        // Handle error
        ATCG_ERROR("Failed to find registerPlugin function in plugin: {0}", path.string());
        FreeLibrary(static_cast<HMODULE>(handle));
        return false;
    }

    using RegisterSystemsFunc = void (*)(ImGuiContext*);
    auto registerSystems =
        reinterpret_cast<RegisterSystemsFunc>(GetProcAddress(static_cast<HMODULE>(handle), "registerSystems"));
    if(!registerSystems)
    {
        ATCG_ERROR("Failed to find registerSystems function in plugin: {0}", path.string());
        FreeLibrary(static_cast<HMODULE>(handle));
        return false;
    }
    registerSystems(ImGui::GetCurrentContext());

    PluginRegistry registry(handle);
    registerPlugin(registry);

    _loaded_plugins[path] = handle;

    return true;
}

bool PluginManagerSystem::releasePlugin(const std::filesystem::path& path)
{
    auto it = _loaded_plugins.find(path);
    if(it != _loaded_plugins.end())
    {
        // Remove all classes registered by this plugin
        MaterialRegistry::Registry* material_registry = MaterialRegistry::getRegistry();
        material_registry->unregisterPlugin(it->second);

        BSDFRegistry::Registry* bsdf_registry = BSDFRegistry::getRegistry();
        bsdf_registry->unregisterPlugin(it->second);


        // TODO Platform-specific implementation for releasing plugin using FreeLibrary (Windows) or dlclose (Linux)
        FreeLibrary(static_cast<HMODULE>(it->second));
        _loaded_plugins.erase(it);
        return true;
    }
    return false;
}

bool PluginManagerSystem::releaseAllPlugins()
{
    bool success = true;
    for(auto& [path, handle]: _loaded_plugins)
    {
        MaterialRegistry::Registry* material_registry = MaterialRegistry::getRegistry();
        material_registry->unregisterPlugin(handle);

        BSDFRegistry::Registry* bsdf_registry = BSDFRegistry::getRegistry();
        bsdf_registry->unregisterPlugin(handle);

        // TODO Platform-specific implementation for releasing plugin using FreeLibrary (Windows) or dlclose (Linux)
        if(!FreeLibrary(static_cast<HMODULE>(handle)))
        {
            success = false;
        }
    }
    _loaded_plugins.clear();
    return success;
}
}    // namespace atcg