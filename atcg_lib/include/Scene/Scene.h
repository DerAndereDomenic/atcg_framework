#pragma once

#include <entt.hpp>
#include <memory>

#include <Core/UUID.h>

namespace atcg
{
class Entity;
class Serializer;

/**
 * @brief A class to model a scene.
 */
class Scene
{
public:
    /**
     * @brief Constructor
     */
    Scene();

    /**
     * @brief Destructor
     */
    ~Scene();

    /**
     * @brief Create a new entity that is part of this scene.
     *
     * @return The entity
     */
    Entity createEntity();

    /**
     * @brief Get an entity by its UUID
     *
     * @param id The UUID of the entity
     * @return The entity
     */
    Entity getEntityByID(UUID id) const;

    /**
     * @brief Get a view of all the entities with the requested components
     *
     * @tparam Components The Components that each entity should have
     *
     * @return A view onto the entities
     */
    template<typename... Components>
    auto getAllEntitiesWith() const
    {
        return _registry.view<Components...>();
    }

private:
    friend class Entity;
    friend class Serializer;
    entt::registry _registry;

    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg