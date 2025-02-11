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
     * @brief Create a new entity that is part of this scene using a handle.
     * The handle is a hint that is used if no other entity with this handle already exists
     *
     * @param handle The entity handle
     * @param uuid The uuid
     * @param name The name of the entity
     *
     * @return The entity
     */
    Entity createEntity(const entt::entity handle, UUID uuid, const std::string& name);

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
     * @brief Removes an entity from the scene
     *
     * @param id The id of the entity
     */
    void removeEntity(UUID id);

    /**
     * @brief Removes an entity from the scene
     *
     * @param entity The entity to remove
     */
    void removeEntity(Entity entity);

    /**
     * @brief Remove all entities
     */
    void removeAllEntites();

private:
    void _updateEntityID(atcg::Entity entity, const UUID old_id, const UUID new_id);
    void _updateEntityName(atcg::Entity entity, const std::string& old_name, const std::string& new_name);

private:
    friend class Entity;
    entt::registry _registry;

    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg