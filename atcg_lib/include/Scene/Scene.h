#pragma once

#include <entt.hpp>

namespace atcg
{
class Entity;

class Scene
{
public:
    Scene() = default;

    ~Scene() = default;

    Entity createEntity();

private:
    friend class Entity;
    entt::registry _registry;
};
}    // namespace atcg