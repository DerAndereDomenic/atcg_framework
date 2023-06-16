#include <Scene/Scene.h>

#include <Scene/Entity.h>

namespace atcg
{
Entity Scene::createEntity()
{
    Entity entity(_registry.create(), this);
    return entity;
}
}    // namespace atcg