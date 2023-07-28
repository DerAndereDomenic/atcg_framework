#pragma once

#include <entt.hpp>

#include <Scene/Scene.h>

namespace atcg
{
/**
 * @brief A lightweight class to model an entity

 */
class Entity
{
public:
    /**
     * @brief Default constructor
     */
    Entity() = default;

    /**
     * @brief Create an entity from an entt handle
     *
     * @param handle The handle
     * @param scene The scene that handles this entity
     */
    Entity(entt::entity handle, Scene* scene);

    /**
     * @brief Add a component to the entity
     *
     * @tparam T The component type
     * @tparam Args The arguments of the Component Constructor
     *
     * @param Args The constructor arguments
     *
     * @return Reference to the created component
     */
    template<typename T, typename... Args>
    T& addComponent(Args&&... args)
    {
        T& component = _scene->_registry.emplace<T>(_entity_handle, std::forward<Args>(args)...);
        return component;
    }

    /**
     * @brief Get a component
     *
     * @tparam T The component type
     *
     * @return A reference to the component
     */
    template<typename T>
    T& getComponent()
    {
        return _scene->_registry.get<T>(_entity_handle);
    }

    /**
     * @brief Check if the Entity holds all components
     *
     * @tparam T The component types
     *
     * @return True if the entity has all of the component
     */
    template<typename... T>
    bool hasComponent()
    {
        return _scene->_registry.all_of<T...>(_entity_handle);
    }

    /**
     * @brief Check if the Entity holds any components
     *
     * @tparam T The component types
     *
     * @return True if the entity has any the component
     */
    template<typename... T>
    bool hasAnyComponent()
    {
        return _scene->_registry.any_of<T...>(_entity_handle);
    }

    /**
     * @brief Remove a component
     *
     * @tparam T The component type
     */
    template<typename T>
    void removeComponent()
    {
        _scene->_registry.remove<T>(_entity_handle);
    }

    /**
     * @brief Check if this is an empty entity
     *
     * @return Whether this is an empty entity
     */
    operator bool() const { return _entity_handle != entt::null; }

private:
    friend class Renderer;
    Scene* _scene               = nullptr;
    entt::entity _entity_handle = entt::null;
};
}    // namespace atcg