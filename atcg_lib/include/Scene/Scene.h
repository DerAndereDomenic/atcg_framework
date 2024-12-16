#pragma once


#include <Core/UUID.h>

#include <memory>
#include <entt.hpp>
namespace atcg
{
class Entity;

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
     * @param name The name of the entity
     *
     * @return The entity
     */
    Entity createEntity(const std::string& name = "Entity");

    /**
     * @brief Get an entity by its UUID.
     * If no entity exists with that ID, an invalid entity is returned
     *
     * @param id The UUID of the entity
     * @return The entity
     */
    Entity getEntityByID(UUID id) const;

    /**
     * @brief Get entities with a given name.
     *
     * @param name The name to search
     *
     * @return Entitites with the given name
     */
    std::vector<Entity> getEntitiesByName(const std::string& name);

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

    /**
     * @brief Remove all entities
     */
    void removeAllEntites() { _registry.clear(); }

private:
    friend class Entity;
    entt::registry _registry;

    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg