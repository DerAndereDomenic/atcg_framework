#pragma once

#include <Core/Platform.h>
#include <Scene/Scene.h>

#include <entt.hpp>

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
     * @brief Set a component to the entity
     *
     * @tparam T The component type
     * @tparam The component to replace
     *
     * @return Reference to the created component
     */
    template<typename T>
    T& replaceComponent(T& component)
    {
        T& comp = _scene->_registry.replace<T>(_entity_handle, component);
        return comp;
    }

    /**
     * @brief Add or replace a component to the entity
     *
     * @tparam T The component type
     * @tparam Args The arguments of the Component Constructor
     *
     * @param Args The constructor arguments
     *
     * @return Reference to the created component
     */
    template<typename T, typename... Args>
    T& addOrReplaceComponent(Args&&... args)
    {
        T& comp = _scene->_registry.emplace_or_replace<T>(_entity_handle, std::forward<Args>(args)...);
        return comp;
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
     * @tparam T... The component types
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
     * @tparam T... The component types
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
     * @brief Get the entity handle.
     * This handle is a unique id for this entity in the scene and is internally used to identify the entity. HOWEVER,
     * this ID is not prersistent for this entity when serializing a scene for example. This ID is only used by the
     * renderer for mouse picking. If you want to identify an entity uniquely and persistently, use the IDComponent that
     * each entity has.
     *
     * This ID can also be used for RendererSystem::draw if it is necessary to have complete fine-grained control over
     * the rendering behavior of an entity. However, in most cases, the RendererSystem::draw for entities or
     * RendererSystem::drawComponent functions should be sufficient.
     *
     * @return The entity handle
     */
    ATCG_INLINE uint32_t entity_handle() const { return (uint32_t)_entity_handle; }

    /**
     * @brief Check if this is an empty entity
     *
     * @return Whether this is an empty entity
     */
    operator bool() const { return (_entity_handle != entt::null && _scene->_registry.valid(_entity_handle)); }

private:
    friend class Scene;
    friend class RendererSystem;
    Scene* _scene               = nullptr;
    entt::entity _entity_handle = entt::null;
};
}    // namespace atcg