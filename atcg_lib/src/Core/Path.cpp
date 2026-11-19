#include <Core/Path.h>

#include <cstdlib>

#if ATCG_PLATFORM_WINDOWS
    #include <Windows.h>
#else
    #include <dlfcn.h>
    #include <unistd.h>
#endif

namespace atcg
{
namespace detail
{
/**
 * @brief Resolve a runtime directory of the framework
 *
 * @param env_variable The environment variable that overrides the directory
 * @param build_directory The directory in the build/source tree
 *
 * @return The path
 */
std::filesystem::path resolveDirectory(const char* env_variable, const char* build_directory)
{
    const char* env_dir = std::getenv(env_variable);
    if(env_dir != nullptr && env_dir[0] != '\0')
    {
        return std::filesystem::path(env_dir);
    }

    return std::filesystem::path(build_directory);
}
}    // namespace detail

std::filesystem::path library_directory()
{
#if ATCG_PLATFORM_WINDOWS
    // The address of this function lies inside of atcg_lib, so the module handle is the one of atcg_lib.dll
    HMODULE module = nullptr;
    if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCWSTR)&library_directory,
                           &module))
    {
        return std::filesystem::path();
    }

    wchar_t buffer[MAX_PATH];
    DWORD length = GetModuleFileNameW(module, buffer, MAX_PATH);
    if(length == 0 || length == MAX_PATH)
    {
        return std::filesystem::path();
    }
    return std::filesystem::path(std::wstring(buffer, length)).parent_path();
#else
    Dl_info info;
    if(dladdr((void*)&library_directory, &info) == 0 || info.dli_fname == nullptr)
    {
        return std::filesystem::path();
    }

    std::error_code error;
    return std::filesystem::canonical(info.dli_fname, error).parent_path();
#endif
}

std::filesystem::path executable_path()
{
#if ATCG_PLATFORM_WINDOWS
    wchar_t buffer[MAX_PATH];
    DWORD length = GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    if(length == 0 || length == MAX_PATH)
    {
        return std::filesystem::path();
    }
    return std::filesystem::path(std::wstring(buffer, length));
#else
    std::error_code error;
    return std::filesystem::read_symlink("/proc/self/exe", error);
#endif
}

std::filesystem::path user_config_directory()
{
    std::filesystem::path base;

#if ATCG_PLATFORM_WINDOWS
    const char* appdata = std::getenv("APPDATA");
    if(appdata != nullptr && appdata[0] != '\0')
    {
        base = appdata;
    }
#else
    const char* xdg_config = std::getenv("XDG_CONFIG_HOME");
    const char* home       = std::getenv("HOME");
    if(xdg_config != nullptr && xdg_config[0] != '\0')
    {
        base = xdg_config;
    }
    else if(home != nullptr && home[0] != '\0')
    {
        base = std::filesystem::path(home) / ".config";
    }
#endif

    if(base.empty())
    {
        // No user directory available, fall back to the working directory
        base = std::filesystem::current_path();
    }

    std::filesystem::path config_dir = base / "atcg";

    std::error_code error;
    std::filesystem::create_directories(config_dir, error);
    if(error)
    {
        ATCG_WARN("Could not create config directory {}: {}", config_dir.string(), error.message());
    }

    return config_dir;
}

std::filesystem::path shader_directory()
{
    return detail::resolveDirectory("ATCG_SHADER_DIR", ATCG_SHADER_DIR);
}

std::filesystem::path resource_directory()
{
    return detail::resolveDirectory("ATCG_RESOURCE_DIR", ATCG_RESOURCE_DIR);
}

std::filesystem::path ptx_directory()
{
    return detail::resolveDirectory("ATCG_PTX_DIR", ATCG_PTX_DIR);
}

}    // namespace atcg
