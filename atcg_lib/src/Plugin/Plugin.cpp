#include <Plugin/Plugin.h>
#include <Core/Path.h>
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

#ifndef ATCG_HEADLESS
    using RegisterSystemsFunc = void (*)(ImGuiContext*);
    auto registerSystems      = detail::getFunction<RegisterSystemsFunc>(handle, "registerSystems");
    if(!registerSystems)
    {
        ATCG_ERROR("Failed to find registerSystems function in plugin: {0}", path.string());
        detail::freeLibrary(handle);
        return false;
    }
    registerSystems(ImGui::GetCurrentContext());
#endif

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

#ifdef ATCG_CUDA_BACKEND
    // The plugin may have added materials without a matching BSDF
    BSDFRegistry::validateMaterialCoverage();
#endif

    _loaded_plugins[path] = handle;

    return true;
}

bool PluginManagerSystem::loadPluginFromName(const std::string& name)
{
#ifdef ATCG_BUILD_DEBUG
    const std::string build_dir_name = "Debug";
#elif defined(ATCG_BUILD_RELEASE)
    const std::string build_dir_name = "Release";
#else
    const std::string build_dir_name = "RelWithDebInfo";
#endif

#if ATCG_PLATFORM_WINDOWS
    const std::string file_name     = name + ".dll";
    const std::string lib_file_name = file_name;
#else
    const std::string file_name      = name + ".so";
    const std::string lib_file_name  = "lib" + name + ".so";    // Default name of shared libraries built by CMake
#endif

    // Plugins next to atcg_lib come first. This ensures that a build of the library (e.g. the python and the C++ build)
    // loads the plugins that were built against it. The working directory is used as a fallback.
    const std::filesystem::path library_dir       = atcg::library_directory();
    const std::filesystem::path candidate_paths[] = {library_dir / file_name,
                                                     library_dir / lib_file_name,
                                                     std::filesystem::path(file_name),
                                                     std::filesystem::path("bin") / file_name,
                                                     std::filesystem::path("bin") / build_dir_name / file_name,
                                                     std::filesystem::path("lib") / file_name,
                                                     std::filesystem::path("lib") / build_dir_name / file_name};

    for(const auto& path: candidate_paths)
    {
        if(std::filesystem::exists(path))
        {
            bool loaded = loadPlugin(path);
            if(loaded)
            {
                return true;
            }
        }
    }

    return false;
}

bool PluginManagerSystem::releasePluginFromName(const std::string& name)
{
    for(const auto& [path, handle]: _loaded_plugins)
    {
        if(path.stem() == name)
        {
            return releasePlugin(path);
        }
    }
    return false;
}

bool PluginManagerSystem::releasePlugin(const std::filesystem::path& path)
{
    auto it = _loaded_plugins.find(path);
    if(it != _loaded_plugins.end())
    {
        // Remove all classes registered by this plugin
        MaterialRegistry::Registry* material_registry = MaterialRegistry::getRegistry();
        material_registry->unregisterPlugin(it->second);

        RenderPassRegistry::Registry* render_pass_registry = RenderPassRegistry::getRegistry();
        render_pass_registry->unregisterPlugin(it->second);

        ComponentRegistry::Registry* component_registry = ComponentRegistry::getRegistry();
        component_registry->unregisterPlugin(it->second);

#ifdef ATCG_CUDA_BACKEND
        BSDFRegistry::Registry* bsdf_registry = BSDFRegistry::getRegistry();
        bsdf_registry->unregisterPlugin(it->second);

        IntegratorRegistry::Registry* integrator_registry = IntegratorRegistry::getRegistry();
        integrator_registry->unregisterPlugin(it->second);
#endif

        detail::freeLibrary(it->second);

        std::filesystem::path temp_path = path;
        temp_path.replace_extension(".temp");
        std::filesystem::remove(temp_path);
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

        RenderPassRegistry::Registry* render_pass_registry = RenderPassRegistry::getRegistry();
        render_pass_registry->unregisterPlugin(handle);

        ComponentRegistry::Registry* component_registry = ComponentRegistry::getRegistry();
        component_registry->unregisterPlugin(handle);

#ifdef ATCG_CUDA_BACKEND
        BSDFRegistry::Registry* bsdf_registry = BSDFRegistry::getRegistry();
        bsdf_registry->unregisterPlugin(handle);

        IntegratorRegistry::Registry* integrator_registry = IntegratorRegistry::getRegistry();
        integrator_registry->unregisterPlugin(handle);
#endif

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