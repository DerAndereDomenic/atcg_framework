#include <Core/Log.h>
#include "spdlog/sinks/stdout_color_sinks.h"

namespace atcg
{
Logger::Logger()
{
    spdlog::set_pattern("%^[%T] %n: %v%$");
    _logger = spdlog::stdout_color_mt("ATCG");
    _logger->set_level(spdlog::level::trace);
}
}    // namespace atcg