#pragma once

#include <entt.hpp>

#include <Scene/Scene.h>

namespace atcg
{
class Entity
{
public:
    Entity() = default;
    Entity(entt::entity handle, Scene* scene);

    template<typename T, typename... Args>
    T& addComponent(Args&&... args)
    {
        T& component = _scene->_registry.emplace<T>(_entity_handle, std::forward<Args>(args)...);
        return component;
    }

    template<typename T>
    T& getComponent()
    {
        return _scene->_registry.get<T>(_entity_handle);
    }

    template<typename T>
    bool hasComponent()
    {
        return _scene->_registry.all_of<T>(_entity_handle);
    }

    operator bool() const { return _entity_handle != entt::null; }

private:
    Scene* _scene               = nullptr;
    entt::entity _entity_handle = entt::null;
};
}    // namespace atcg