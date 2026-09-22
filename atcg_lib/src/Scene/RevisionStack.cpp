#include <Scene/RevisionStack.h>
#include <Scene/ComponentRegistry.h>

#include <Scene/Components/IDComponent.h>
#include <Scene/Components/NameComponent.h>

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

void EntityAddedRevision::record_start_state()
{
    atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
    _uuid = entity.getComponent<IDComponent>().ID();
    _name = entity.getComponent<NameComponent>().name();
}

void EntityRemovedRevision::record_start_state()
{
    atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
    _uuid = entity.getComponent<IDComponent>().ID();
    _name = entity.getComponent<NameComponent>().name();

    storeComponents(entity);
}

}    // namespace atcg