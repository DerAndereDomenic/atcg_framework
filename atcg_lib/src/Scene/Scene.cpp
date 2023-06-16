#include <Scene/Scene.h>

#include <Scene/Entity.h>
#include <Scene/Components.h>

namespace atcg
{
Entity Scene::createEntity()
{
    Entity entity(_registry.create(), this);
    entity.addComponent<IDComponent>();
    return entity;
}
}    // namespace atcg