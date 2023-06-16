#include <Scene/Entity.h>

namespace atcg
{
Entity::Entity(entt::entity handle, Scene* scene) : _entity_handle(handle), _scene(scene) {}
}    // namespace atcg