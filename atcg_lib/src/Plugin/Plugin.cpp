#include <Plugin/Plugin.h>
#if ATCG_PLATFORM_WINDOWS
    #include <Windows.h>    // TODO
#else
    #include <dlfcn.h>
#endif
#include <pybind11/pybind11.h>

namespace atcg
{

namespace detail
{
PluginHandle loadLibrary(const std::filesystem::path& path)
{
#if ATCG_PLATFORM_WINDOWS
    return (PluginHandle)LoadLibraryA(path.string().c_str());
#else
    return (PluginHandle)dlopen(path.string().c_str(), RTLD_NOW);
#endif
}

template<typename Func>
Func getFunction(PluginHandle handle, const char* name)
{
#if ATCG_PLATFORM_WINDOWS
    return (Func)(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
    return (Func)(dlsym((void*)handle, name));
#endif
}

bool freeLibrary(PluginHandle handle)
{
#if ATCG_PLATFORM_WINDOWS
    return FreeLibrary(static_cast<HMODULE>(handle));
#else
    return (bool)dlclose((void*)handle);
#endif
}
}    // namespace detail

PluginManagerSystem::~PluginManagerSystem()
{
    releaseAllPlugins();
}

bool PluginManagerSystem::loadPlugin(const std::filesystem::path& path)
{
    // Create temp copy of dll at path_temp to avoid file locking issues when reloading the plugin
    std::filesystem::path temp_path = path;
    temp_path.replace_extension(".temp");
    std::filesystem::copy(path, temp_path, std::filesystem::copy_options::overwrite_existing);

    // TODO Platform-specific implementation for loading plugin using LoadLibrary (Windows) or dlopen (Linux)
    // Implementation for loading plugin

    auto handle = detail::loadLibrary(temp_path.string().c_str());
    if(!handle)
    {
        // Handle error
        ATCG_ERROR("Failed to load plugin: {0}", path.string());
        return false;
    }

    using RegisterPluginFunc = void (*)(PluginRegistry&);
    auto registerPlugin      = detail::getFunction<RegisterPluginFunc>(handle, "registerPlugin");
    if(!registerPlugin)
    {
        // Handle error
        ATCG_ERROR("Failed to find registerPlugin function in plugin: {0}", path.string());
        detail::freeLibrary(handle);
        return false;
    }

    using RegisterSystemsFunc = void (*)(ImGuiContext*);
    auto registerSystems      = detail::getFunction<RegisterSystemsFunc>(handle, "registerSystems");
    if(!registerSystems)
    {
        ATCG_ERROR("Failed to find registerSystems function in plugin: {0}", path.string());
        detail::freeLibrary(handle);
        return false;
    }
    registerSystems(ImGui::GetCurrentContext());

    using RegisterPythonBindingsFunc = void (*)(pybind11::module&);
    auto registerPythonBindings = detail::getFunction<RegisterPythonBindingsFunc>(handle, "registerPythonBindings");
    if(registerPythonBindings)
    {
        try
        {
            auto pyatcg = pybind11::module_::import("pyatcg");
            registerPythonBindings(pyatcg);
        }
        catch(const pybind11::error_already_set& e)
        {
            ATCG_ERROR(e.what());
        }
        catch(const std::exception& e)
        {
            ATCG_ERROR(e.what());
        }
    }

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

#ifndef ATCG_CUDA_BACKEND
        BSDFRegistry::Registry* bsdf_registry = BSDFRegistry::getRegistry();
        bsdf_registry->unregisterPlugin(it->second);

        IntegratorRegistry::Registry* integrator_registry = IntegratorRegistry::getRegistry();
        integrator_registry->unregisterPlugin(it->second);
#endif


        // TODO Platform-specific implementation for releasing plugin using FreeLibrary (Windows) or dlclose (Linux)
        detail::freeLibrary(it->second);
        _loaded_plugins.erase(it);

        std::filesystem::path temp_path = path;
        temp_path.replace_extension(".temp");
        std::filesystem::remove(temp_path);

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

#ifdef ATCG_CUDA_BACKEND
        BSDFRegistry::Registry* bsdf_registry = BSDFRegistry::getRegistry();
        bsdf_registry->unregisterPlugin(handle);

        IntegratorRegistry::Registry* integrator_registry = IntegratorRegistry::getRegistry();
        integrator_registry->unregisterPlugin(handle);
#endif

        // TODO Platform-specific implementation for releasing plugin using FreeLibrary (Windows) or dlclose (Linux)
        if(!detail::freeLibrary(handle))
        {
            success = false;
        }

        std::filesystem::path temp_path = path;
        temp_path.replace_extension(".temp");
        std::filesystem::remove(temp_path);
    }
    _loaded_plugins.clear();
    return success;
}
}    // namespace atcg