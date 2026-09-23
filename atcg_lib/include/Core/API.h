#pragma once

// Based on Hazel Engine (https://github.com/TheCherno/Hazel)
#define ATCG_BIND_EVENT_FN(fn)                                                                                         \
    [this](auto&&... args) -> decltype(auto)                                                                           \
    {                                                                                                                  \
        return this->fn(std::forward<decltype(args)>(args)...);                                                        \
    }

// -----------------------------
// Platform detection
// -----------------------------
#if defined(_WIN32) || defined(_WIN64)
    #define ATCG_PLATFORM_WINDOWS 1
#else
    #define ATCG_PLATFORM_WINDOWS 0
#endif

#if defined(__GNUC__) || defined(__clang__)
    #define ATCG_COMPILER_GCC_OR_CLANG 1
#else
    #define ATCG_COMPILER_GCC_OR_CLANG 0
#endif

// -----------------------------
// Shared / static build toggle
// -----------------------------
#if ATCG_PLATFORM_WINDOWS
    #define ATCG_IMPORT __declspec(dllimport)
    #define ATCG_EXPORT __declspec(dllexport)
    #if defined(ATCG_EXPORT_DLL)
        #define ATCG_API ATCG_EXPORT
        #define ATCG_LOCAL
    #else
        #define ATCG_API ATCG_IMPORT
        #define ATCG_LOCAL
    #endif
#else
    // Linux / macOS
    #define ATCG_IMPORT __attribute__((visibility("default")))
    #define ATCG_EXPORT __attribute__((visibility("default")))

    // Hidden by default improves compile times and symbol cleanliness
    #if ATCG_COMPILER_GCC_OR_CLANG
        #define ATCG_API   __attribute__((visibility("default")))
        #define ATCG_LOCAL __attribute__((visibility("hidden")))
    #else
        #define ATCG_API
        #define ATCG_LOCAL
    #endif
#endif