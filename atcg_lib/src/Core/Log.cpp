#include <Core/Log.h>
#include "spdlog/sinks/stdout_color_sinks.h"

namespace atcg
{
atcg::ref_ptr<spdlog::logger> Log::s_logger;

void Log::init()
{
    spdlog::set_pattern("%^[%T] %n: %v%$");
    s_logger = spdlog::stdout_color_mt("MURMELSPIEL");
    s_logger->set_level(spdlog::level::trace);
}
}