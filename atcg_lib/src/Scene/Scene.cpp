#include <Scene/Scene.h>

#include <Scene/Entity.h>
#include <Scene/Components.h>

namespace atcg
{
Entity Scene::createEntity()
{
    Entity entity(_registry.create(), this);
    IDComponent id = entity.addComponent<IDComponent>();

    _entities.insert(std::make_pair(id.ID, entity));
    return entity;
}

Entity Scene::getEntityByID(UUID id) const
{
    return _entities.find(id)->second;
}
}    // namespace atcg