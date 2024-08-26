#include <Core/SystemRegistry.h>
#include <Core/Assert.h>

namespace atcg
{

static SystemRegistry* s_registry = nullptr;

void SystemRegistry::init()
{
    ATCG_ASSERT(!s_registry, "Registry already initialized");

    s_registry = new SystemRegistry;
}

SystemRegistry* SystemRegistry::instance()
{
    ATCG_ASSERT(s_registry, "Registry not initialized");

    return s_registry;
}

void SystemRegistry::setInstance(SystemRegistry* registry)
{
    ATCG_ASSERT(!s_registry, "Registry already exists");
    ATCG_ASSERT(registry, "Must be a valid registry instance");

    s_registry = registry;
}

void SystemRegistry::shutdown()
{
    ATCG_ASSERT(s_registry, "Registry not initialized");

    delete s_registry;
    s_registry = nullptr;
}
}    // namespace atcg