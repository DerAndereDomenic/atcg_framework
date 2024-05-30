#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include <Core/Platform.h>

namespace atcg
{
/**
 * @brief A class to handle logging
 */
class Log
{
public:
    /**
     * @brief Initialize the logger
     */
    static void init();

    /**
     * @brief Get the logger
     *
     * @return The logger
     *
     */
    ATCG_INLINE static std::shared_ptr<spdlog::logger>& getLogger() { return s_logger; }

private:
    static std::shared_ptr<spdlog::logger> s_logger;
};
}    // namespace atcg

#ifdef NDEBUG
    #define ATCG_ERROR(...) atcg::Log::getLogger()->error(__VA_ARGS__)
    #define ATCG_WARN(...)
    #define ATCG_INFO(...) atcg::Log::getLogger()->info(__VA_ARGS__)
    #define ATCG_TRACE(...)
#else
    #define ATCG_ERROR(...) atcg::Log::getLogger()->error(__VA_ARGS__)
    #define ATCG_WARN(...)  atcg::Log::getLogger()->warn(__VA_ARGS__)
    #define ATCG_INFO(...)  atcg::Log::getLogger()->info(__VA_ARGS__)
    #define ATCG_TRACE(...) atcg::Log::getLogger()->trace(__VA_ARGS__)
#endif

#define ATCG_LOG_ALLOCATION(...) ATCG_TRACE(__VA_ARGS__)