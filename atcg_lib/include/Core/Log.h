#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include <Core/Platform.h>
#include <Core/SystemRegistry.h>

namespace atcg
{
/**
 * @brief A class to handle logging
 */
class Logger
{
public:
    /**
     * @brief Initialize the logger
     */
    Logger();

    /**
     * @brief Get the logger
     *
     * @return The logger
     */
    ATCG_INLINE std::shared_ptr<spdlog::logger>& getLogger() { return _logger; }

private:
    std::shared_ptr<spdlog::logger> _logger;
};
}    // namespace atcg

#ifdef NDEBUG
    #define ATCG_ERROR(...) atcg::SystemRegistry::instance()->getSystem<atcg::Logger>()->getLogger()->error(__VA_ARGS__)
    #define ATCG_WARN(...)
    #define ATCG_INFO(...) atcg::SystemRegistry::instance()->getSystem<atcg::Logger>()->getLogger()->info(__VA_ARGS__)
    #define ATCG_TRACE(...)
#else
    #define ATCG_ERROR(...) atcg::SystemRegistry::instance()->getSystem<atcg::Logger>()->getLogger()->error(__VA_ARGS__)
    #define ATCG_WARN(...)  atcg::SystemRegistry::instance()->getSystem<atcg::Logger>()->getLogger()->warn(__VA_ARGS__)
    #define ATCG_INFO(...)  atcg::SystemRegistry::instance()->getSystem<atcg::Logger>()->getLogger()->info(__VA_ARGS__)
    #define ATCG_TRACE(...) atcg::SystemRegistry::instance()->getSystem<atcg::Logger>()->getLogger()->trace(__VA_ARGS__)
#endif

#if 0
    #define ATCG_LOG_ALLOCATION(...) ATCG_TRACE(__VA_ARGS__)
#else
    #define ATCG_LOG_ALLOCATION(...)
#endif