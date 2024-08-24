#include <Core/SystemRegistry.h>

namespace atcg
{

static SystemRegistry* s_registry = nullptr;

void SystemRegistry::init()
{
    s_registry = new SystemRegistry;
}

SystemRegistry* SystemRegistry::instance()
{
    return s_registry;
}

void SystemRegistry::setInstance(SystemRegistry* registry)
{
    s_registry = registry;
}

void SystemRegistry::shutdown()
{
    delete s_registry;
    s_registry = nullptr;
}
}    // namespace atcg