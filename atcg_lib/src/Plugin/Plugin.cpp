#include <Plugin/Plugin.h>
#include <Windows.h>    // TODO

namespace atcg
{

PluginManager::~PluginManager()
{
    _registered_classes.clear();
    for(auto& [path, handle]: _loaded_plugins)
    {
        FreeLibrary(static_cast<HMODULE>(handle));    // TODO Platform-specific implementation for releasing plugin
                                                      // using FreeLibrary (Windows) or dlclose (Linux)
    }
    _loaded_plugins.clear();
}

bool PluginManager::loadPlugin(const std::filesystem::path& path)
{
    // TODO Platform-specific implementation for loading plugin using LoadLibrary (Windows) or dlopen (Linux)
    // Implementation for loading plugin

    auto handle = LoadLibraryA(path.string().c_str());
    if(!handle)
    {
        // Handle error
        return false;
    }

    using RegisterPluginFunc = void (*)(PluginRegistry&);
    auto registerPlugin =
        reinterpret_cast<RegisterPluginFunc>(GetProcAddress(static_cast<HMODULE>(handle), "registerPlugin"));
    if(!registerPlugin)
    {
        // Handle error
        FreeLibrary(static_cast<HMODULE>(handle));
        return false;
    }

    PluginRegistry registry(*this, handle);
    registerPlugin(registry);

    _loaded_plugins[path] = handle;

    return true;
}

bool PluginManager::releasePlugin(const std::filesystem::path& path)
{
    auto it = _loaded_plugins.find(path);
    if(it != _loaded_plugins.end())
    {
        // Remove all classes registered by this plugin
        for(auto it_class = _registered_classes.begin(); it_class != _registered_classes.end();)
        {
            if(it_class->second->library_handle == it->second)
            {
                it_class = _registered_classes.erase(it_class);
            }
            else
            {
                ++it_class;
            }
        }

        // TODO Platform-specific implementation for releasing plugin using FreeLibrary (Windows) or dlclose (Linux)
        FreeLibrary(static_cast<HMODULE>(it->second));
        _loaded_plugins.erase(it);
        return true;
    }
    return false;
}
}    // namespace atcg