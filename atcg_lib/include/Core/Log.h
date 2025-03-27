#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include <Core/Platform.h>
#include <Core/SystemRegistry.h>
#include "spdlog/sinks/stdout_color_sinks.h"

#define ATCG_ERROR(...) atcg::SystemRegistry::instance()->getSystem<spdlog::logger>()->error(__VA_ARGS__)
#define ATCG_WARN(...)  atcg::SystemRegistry::instance()->getSystem<spdlog::logger>()->warn(__VA_ARGS__)
#define ATCG_INFO(...)  atcg::SystemRegistry::instance()->getSystem<spdlog::logger>()->info(__VA_ARGS__)
#define ATCG_TRACE(...) atcg::SystemRegistry::instance()->getSystem<spdlog::logger>()->trace(__VA_ARGS__)
#define ATCG_DEBUG(...) atcg::SystemRegistry::instance()->getSystem<spdlog::logger>()->debug(__VA_ARGS__)

#if 0
    #define ATCG_LOG_ALLOCATION(...) ATCG_TRACE(__VA_ARGS__)
#else
    #define ATCG_LOG_ALLOCATION(...)
#endif