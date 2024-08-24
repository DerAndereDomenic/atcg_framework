#include <Core/SystemRegistry.h>

namespace atcg
{

static const SystemRegistry* s_registry = nullptr;

void SystemRegistry::init()
{
    s_registry = new SystemRegistry;
}

const SystemRegistry* SystemRegistry::instance()
{
    return s_registry;
}

void SystemRegistry::setInstance(const SystemRegistry* registry)
{
    s_registry = registry;
}

void SystemRegistry::shutdown()
{
    delete s_registry;
    s_registry = nullptr;
}
}    // namespace atcg