#pragma once

#include <Core/Memory.h>
#include <Scene/Entity.h>

#include <memory>

namespace atcg
{
class Scene;

/**
 * @brief A template struct to copy components (used for copy & paste of entities, see Clipboard).
 * @tparam T The component type
 *
 * The default implementation uses the copy constructor of the component. Components that own resources which must not
 * be shared between the original and the copy (cameras, buffers, behaviors, ...) specialize this struct. Assets that
 * are referenced by handle (materials, graphs, ...) are shared on purpose.
 *
 * @code{.cpp}
 * template<>
 * struct atcg::ComponentCopier<CustomComponent>
 * {
 *     static CustomComponent copy(const CustomComponent& component);
 *     static void paste(const atcg::ref_ptr<Scene>& scene, Entity entity, const CustomComponent& component);
 * };
 * @endcode
 */
template<typename T>
struct ComponentCopier
{
    /**
     * @brief Create an independent copy of the component
     *
     * @param component The component
     *
     * @return The copy
     */
    static T copy(const T& component) { return component; }

    /**
     * @brief Add a copy of the component to an entity
     *
     * @param scene The scene of the entity
     * @param entity The entity that receives the component
     * @param component The component to paste
     */
    static void paste(const atcg::ref_ptr<Scene>& scene, Entity entity, const T& component)
    {
        entity.addOrReplaceComponent<T>(copy(component));
    }
};

#define ATCG_DECLARE_COMPONENT_COPIER(ComponentType)                                                                   \
    template<>                                                                                                         \
    struct ATCG_API ComponentCopier<ComponentType>                                                                     \
    {                                                                                                                  \
        static ComponentType copy(const ComponentType& component);                                                     \
        static void paste(const atcg::ref_ptr<Scene>& scene, Entity entity, const ComponentType& component);           \
    };

/**
 * @brief Copy a type erased component (see ComponentRegistry::copyAllComponents)
 *
 * @tparam T The component type
 * @param id The id of the component type (entt::type_hash)
 * @param component The component
 *
 * @return The copy or nullptr if the component is not of type T
 */
template<typename T>
std::shared_ptr<void> copyComponent(const entt::id_type id, const std::shared_ptr<void>& component)
{
    if(id != entt::type_hash<T>::value()) return nullptr;

    return std::make_shared<T>(ComponentCopier<T>::copy(*std::static_pointer_cast<T>(component)));
}

/**
 * @brief Paste a type erased component onto an entity (see ComponentRegistry::pasteAllComponents)
 *
 * @tparam T The component type
 * @param scene The scene of the entity
 * @param entity The entity
 * @param id The id of the component type (entt::type_hash)
 * @param component The component
 */
template<typename T>
void pasteComponent(const atcg::ref_ptr<Scene>& scene,
                    Entity entity,
                    const entt::id_type id,
                    const std::shared_ptr<void>& component)
{
    if(id != entt::type_hash<T>::value()) return;

    ComponentCopier<T>::paste(scene, entity, *std::static_pointer_cast<T>(component));
}
}    // namespace atcg
