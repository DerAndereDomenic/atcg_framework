#pragma once

#include <entt.hpp>
#include <unordered_map>

#include <Core/UUID.h>

namespace atcg
{
class Entity;

class Scene
{
public:
    Scene() = default;

    ~Scene() = default;

    Entity createEntity();

    Entity getEntityByID(UUID id) const;

    template<typename... Components>
    auto getAllEntitiesWith() const
    {
        return _registry.view<Components...>();
    }

private:
    friend class Entity;
    entt::registry _registry;

    std::unordered_map<UUID, Entity> _entities;
};
}    // namespace atcg