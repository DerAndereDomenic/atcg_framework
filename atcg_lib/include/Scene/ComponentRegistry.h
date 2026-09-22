#pragma once

#include <Asset/Asset.h>
#include <Core/API.h>
#include <Core/Assert.h>
#include <DataStructure/Dictionary.h>
#include <DataStructure/Registry.h>
#include <Scene/Entity.h>
#include <Renderer/Renderer.h>
#include <Scene/ComponentRenderer.h>
#include <Scene/ComponentGUIHandler.h>
#include <Scene/ComponentSerializer.h>

#include <json.hpp>
#include <filesystem>


namespace atcg
{
namespace ComponentRegistry    // !
{
using SerializeComponentFn =
    std::function<void(const std::string&, const atcg::ref_ptr<Scene>&, Entity, nlohmann::json&)>;
using DeserializeComponentFn =
    std::function<void(const std::string&, const atcg::ref_ptr<Scene>&, Entity, nlohmann::json&)>;
using DrawComponentFn       = std::function<void(const atcg::ref_ptr<Scene>&, Entity)>;
using DisplayAddComponentFn = std::function<void(const atcg::ref_ptr<Scene>&, Entity)>;
using StoreComponentFn      = std::function<void(Entity, std::unordered_map<entt::id_type, std::shared_ptr<void>>&)>;
using RestoreComponentFn    = std::function<void(Entity, const entt::id_type, const std::shared_ptr<void>&)>;
using RenderComponentFn = std::function<void(RendererSystem*, Entity, const atcg::ref_ptr<Camera>&, atcg::Dictionary&)>;

struct ComponentFunctions
{
    SerializeComponentFn serialize;
    DeserializeComponentFn deserialize;
    DrawComponentFn draw;
    DisplayAddComponentFn display_add;
    StoreComponentFn store;
    RestoreComponentFn restore;
    RenderComponentFn render;
};

using Registry = atcg::Registry<ComponentFunctions>;

void registerComponent(Registry* registry, std::string_view type, ComponentFunctions functions);

/**
 * @brief Serializes all registered components of an entity
 *
 * @param registry The component registry
 * @param file_path The file path of the serialized file
 * @param scene The scene where the entity belongs to
 * @param entity The entity that is serialized
 * @param j The json object describing this entity
 */
void serializeAllComponents(Registry* registry,
                            const std::string& file_path,
                            const atcg::ref_ptr<Scene>& scene,
                            Entity entity,
                            nlohmann::json& j);

/**
 * @brief Deserializes all registered components of an entity
 *
 * @param registry The component registry
 * @param file_path The file path of the serialized file
 * @param scene The scene where the entity belongs to
 * @param entity The entity that is serialized
 * @param j The json object describing this entity
 */
void deserializeAllComponents(Registry* registry,
                              const std::string& file_path,
                              const atcg::ref_ptr<Scene>& scene,
                              Entity entity,
                              nlohmann::json& j);

/**
 * @brief Draws all registered components of an entity in the SceneHierarchyPanel
 *
 * @param registry The component registry
 * @param scene The scene where the entity belongs to
 * @param entity The entity to display
 */
void drawAllComponents(Registry* registry, const atcg::ref_ptr<Scene>& scene, Entity entity);

/**
 * @brief Display components in the "Add Component" Drop down menu
 *
 * @param registry The component registry
 * @param scene The scene where the entity belongs to
 * @param entity The entity to display
 */
void displayAddAllComponents(Registry* registry, const atcg::ref_ptr<Scene>& scene, Entity entity);

/**
 * @brief This function is used by the RevisionSystem to create a copy of the entity and its components
 *
 * @param registry The component registry
 * @param entity The entity to copy
 * @param components The output components stored by the entity as generic pointers
 */
void storeAllComponents(Registry* registry,
                        Entity entity,
                        std::unordered_map<entt::id_type, std::shared_ptr<void>>& components);

/**
 * @brief Restores and entity with its components
 *
 * @param registry The component registry
 * @param entity The entity
 * @param id The id of the Component type (entt::hash)
 * @param component The component as generic pointer
 */
void restoreAddAllComponents(Registry* registry,
                             Entity entity,
                             const entt::id_type id,
                             const std::shared_ptr<void>& component);

/**
 * @brief Render all components
 *
 * @param registry The component registry
 * @param renderer The renderer
 * @param entity The entity
 * @param camera The camera
 * @param auxiliary Dictionary with auxiliary information
 */
void renderAllComponents(Registry* registry,
                         RendererSystem* renderer,
                         Entity entity,
                         const atcg::ref_ptr<Camera>& camera,
                         atcg::Dictionary& auxiliary);

ATCG_INLINE Registry* getRegistry()
{
    Registry* registry = SystemRegistry::instance()->getSystem<ComponentRegistry::Registry>();
    ATCG_ASSERT(registry, "Component registry not found");
    return registry;
}

/**
 * @brief Serializes all registered components of an entity
 *
 * @param file_path The file path of the serialized file
 * @param scene The scene where the entity belongs to
 * @param entity The entity that is serialized
 * @param j The json object describing this entity
 */
ATCG_INLINE void serializeAllComponents(const std::string& file_path,
                                        const atcg::ref_ptr<Scene>& scene,
                                        Entity entity,
                                        nlohmann::json& j)
{
    serializeAllComponents(getRegistry(), file_path, scene, entity, j);
}

/**
 * @brief Deserializes all registered components of an entity
 *
 * @param file_path The file path of the serialized file
 * @param scene The scene where the entity belongs to
 * @param entity The entity that is serialized
 * @param j The json object describing this entity
 */
ATCG_INLINE void deserializeAllComponents(const std::string& file_path,
                                          const atcg::ref_ptr<Scene>& scene,
                                          Entity entity,
                                          nlohmann::json& j)
{
    deserializeAllComponents(getRegistry(), file_path, scene, entity, j);
}

/**
 * @brief Draws all registered components of an entity in the SceneHierarchyPanel
 *
 * @param scene The scene where the entity belongs to
 * @param entity The entity to display
 */
ATCG_INLINE void drawAllComponents(const atcg::ref_ptr<Scene>& scene, Entity entity)
{
    drawAllComponents(getRegistry(), scene, entity);
}

/**
 * @brief Display components in the "Add Component" Drop down menu
 *
 * @param scene The scene where the entity belongs to
 * @param entity The entity to display
 */
ATCG_INLINE void displayAddAllComponents(const atcg::ref_ptr<Scene>& scene, Entity entity)
{
    displayAddAllComponents(getRegistry(), scene, entity);
}

/**
 * @brief This function is used by the RevisionSystem to create a copy of the entity and its components
 *
 * @param entity The entity to copy
 * @param components The output components stored by the entity as generic pointers
 */
ATCG_INLINE void storeAllComponents(Entity entity, std::unordered_map<entt::id_type, std::shared_ptr<void>>& components)
{
    storeAllComponents(getRegistry(), entity, components);
}

/**
 * @brief Restores and entity with its components
 *
 * @param entity The entity
 * @param id The id of the Component type (entt::hash)
 * @param component The component as generic pointer
 */
ATCG_INLINE void restoreAddAllComponents(Entity entity, const entt::id_type id, const std::shared_ptr<void>& component)
{
    restoreAddAllComponents(getRegistry(), entity, id, component);
}

/**
 * @brief Render all components
 *
 * @param renderer The renderer
 * @param entity The entity
 * @param camera The camera
 * @param auxiliary Dictionary with auxiliary information
 */
ATCG_INLINE void renderAllComponents(RendererSystem* renderer,
                                     Entity entity,
                                     const atcg::ref_ptr<Camera>& camera,
                                     atcg::Dictionary& auxiliary)
{
    renderAllComponents(getRegistry(), renderer, entity, camera, auxiliary);
}
}    // namespace ComponentRegistry
}    // namespace atcg

#define ATCG_REGISTER_COMPONENT(registry, ComponentType, ComponentClass)                                               \
    {                                                                                                                  \
        auto serialize_fn   = &atcg::Serialization::serializeComponent<ComponentClass>;                                \
        auto deserialize_fn = &atcg::Serialization::deserializeComponent<ComponentClass>;                              \
        auto draw_fn        = &atcg::GUI::drawComponent<ComponentClass>;                                               \
        auto display_add_fn = &atcg::GUI::displayAddComponentEntry<ComponentClass>;                                    \
        auto store_fn       = &atcg::storeComponent<ComponentClass>;                                                   \
        auto restore_fn     = &atcg::restoreComponent<ComponentClass>;                                                 \
        auto render_fn      = &atcg::renderComponent<ComponentClass>;                                                  \
        atcg::ComponentRegistry::ComponentFunctions                                                                    \
            functions {serialize_fn, deserialize_fn, draw_fn, display_add_fn, store_fn, restore_fn, render_fn};        \
        registry->registerType(ComponentType, std::move(functions));                                                   \
    }

#define ATCG_REGISTER_COMPONENT_PLUGIN(registry, handle, ComponentType, ComponentClass)                                \
    {                                                                                                                  \
        auto serialize_fn   = &atcg::Serialization::serializeComponent<ComponentClass>;                                \
        auto deserialize_fn = &atcg::Serialization::deserializeComponent<ComponentClass>;                              \
        auto draw_fn        = &atcg::GUI::drawComponent<ComponentClass>;                                               \
        auto display_add_fn = &atcg::GUI::displayAddComponentEntry<ComponentClass>;                                    \
        auto store_fn       = &atcg::storeComponent<ComponentClass>;                                                   \
        auto restore_fn     = &atcg::restoreComponent<ComponentClass>;                                                 \
        auto render_fn      = &atcg::renderComponent<ComponentClass>;                                                  \
        atcg::ComponentRegistry::ComponentFunctions                                                                    \
            functions {serialize_fn, deserialize_fn, draw_fn, display_add_fn, store_fn, restore_fn, render_fn};        \
        registry->registerType(handle, ComponentType, std::move(functions));                                           \
    }