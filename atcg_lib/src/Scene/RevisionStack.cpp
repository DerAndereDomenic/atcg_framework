#include <Scene/RevisionStack.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{

// These functions are implemented in a cpp unit because they rely on the ComponentRegistry which also relies on the
// RevisionStack and therefore, we get a circular dependency.
void EntityRemovedRevision::storeComponents(atcg::Entity entity)
{
    // Capture all components
    ComponentRegistry::storeAllComponents(entity, _components);
}

void EntityRemovedRevision::restoreComponents(atcg::Entity entity)
{
    // Restore components
    for(auto& [id, component]: _components)
    {
        ComponentRegistry::restoreAddAllComponents(entity, id, component);
    }
}
}    // namespace atcg