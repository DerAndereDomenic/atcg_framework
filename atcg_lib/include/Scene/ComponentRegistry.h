#pragma once

#include <Core/API.h>
#include <Core/SystemRegistry.h>
#include <Scene/Entity.h>
#include <Scene/ComponentSerializer.h>
#include <Scene/ComponentGUIHandler.h>
#include <Scene/ComponentRenderer.h>

#include <json.hpp>
#include <vector>

namespace atcg
{

using serialize_fn   = std::function<void(const std::string&, const atcg::ref_ptr<Scene>&, Entity, nlohmann::json&)>;
using deserialize_fn = std::function<void(const std::string&, const atcg::ref_ptr<Scene>&, Entity, nlohmann::json&)>;
using draw_fn        = std::function<void(const atcg::ref_ptr<Scene>&, Entity)>;
using display_add_fn = std::function<void(const atcg::ref_ptr<Scene>&, Entity)>;
using store_fn       = std::function<void(Entity, std::unordered_map<entt::id_type, std::shared_ptr<void>>&)>;
using restore_fn     = std::function<void(Entity, const entt::id_type, const std::shared_ptr<void>&)>;
using render_fn      = std::function<void(RendererSystem*, Entity, const atcg::ref_ptr<Camera>&, atcg::Dictionary&)>;

struct ComponentSerializationEntry
{
    ComponentSerializationEntry(const serialize_fn& serialize, const deserialize_fn& deserialize)
        : serialize(serialize),
          deserialize(deserialize)
    {
    }

    serialize_fn serialize;
    deserialize_fn deserialize;
};

struct ComponentDrawEntry
{
    ComponentDrawEntry(const draw_fn& draw, const display_add_fn& display_add) : draw(draw), display_add(display_add) {}

    draw_fn draw;
    display_add_fn display_add;
};

struct ComponentStoreEntry
{
    ComponentStoreEntry(const store_fn& store, const restore_fn& restore) : store(store), restore(restore) {}

    store_fn store;
    restore_fn restore;
};

struct ComponentRenderEntry
{
    ComponentRenderEntry(const render_fn& render) : render(render) {}

    render_fn render;
};

ATCG_INLINE std::vector<ComponentSerializationEntry>& getSerializationEntries()
{
    static std::vector<ComponentSerializationEntry> entries;
    return entries;
}

ATCG_INLINE std::vector<ComponentDrawEntry>& getDrawEntries()
{
    static std::vector<ComponentDrawEntry> entries;
    return entries;
}

ATCG_INLINE std::vector<ComponentStoreEntry>& getStoreEntries()
{
    static std::vector<ComponentStoreEntry> entries;
    return entries;
}

ATCG_INLINE std::vector<ComponentRenderEntry>& getRenderEntries()
{
    static std::vector<ComponentRenderEntry> entries;
    return entries;
}

/**
 * @brief A class to handle registered components.
 * Components don't need to be registered to be used with entities. However, if your component should be visible in the
 * SceneHierarchyPanel, affected by the Revision System, or be serializable, then you have to implement the
 * corresponding template specializations and regsiter the component so it is visible to those system. An example could
 * look like this:
 *
 * @code{.cpp}
 * struct CustomComponent
 * {
 *     int x = 5;
 *
 *     static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Custom"; }
 * };
 *
 * template<>
 * struct atcg::Serialization::ComponentSerializer<CustomComponent>
 * {
 *     void serialize_component(const std::string& file_path,
 *                              const atcg::ref_ptr<atcg::Scene>& scene,
 *                              atcg::Entity entity,
 *                              CustomComponent& component,
 *                              nlohmann::json& j) const
 *     {
 *         j["Custom"] = component.x;
 *     }
 *
 *     void deserialize_component(const std::string& file_path,
 *                                const atcg::ref_ptr<atcg::Scene>& scene,
 *                                atcg::Entity entity,
 *                                nlohmann::json& j) const
 *     {
 *         if(j.contains("Custom"))
 *         {
 *             auto& component = entity.addComponent<CustomComponent>();
 *             component.x     = int(j["Custom"]);
 *         }
 *     }
 * };
 *
 * template<>
 * struct atcg::GUI::ComponentGUIRenderer<CustomComponent>
 * {
 *     void draw_component(const atcg::ref_ptr<atcg::Scene>& scene, Entity entity, CustomComponent& component) const
 *     {
 *         int value    = component.x;
 *         bool updated = ImGui::SliderInt("Custom", &value, 0, 10);
 *
 *         if(updated)
 *         {
 *             atcg::RevisionStack::startRecording<atcg::ComponentEditedRevision<CustomComponent>>(scene, entity);
 *             component.x = value;
 *             atcg::RevisionStack::endRecording();
 *         }
 *     }
 * };
 *
 * template<>
 * struct atcg::ComponentRenderer<CustomComponent>
 * {
 *     void renderComponent(atcg::RendererSystem* renderer,
 *                          Entity entity,
 *                          const atcg::ref_ptr<Camera>& camera,
 *                          atcg::Dictionary& auxiliary) const
 *     {
 *         renderer->drawCircle(glm::vec3(0), 1.0f, 0.2f, glm::vec3(1), camera);
 *     }
 * };
 *
 * namespace atcg
 * {
 * ATCG_REGISTER_COMPONENT(CustomComponent);
 * }
 * @endcode
 */
class ATCG_API ComponentRegistrySystem
{
public:
    /**
     * @brief Serializes all registered components of an entity
     *
     * @param file_path The file path of the serialized file
     * @param scene The scene where the entity belongs to
     * @param entity The entity that is serialized
     * @param j The json object describing this entity
     */
    void serializeAllComponents(const std::string& file_path,
                                const atcg::ref_ptr<Scene>& scene,
                                Entity entity,
                                nlohmann::json& j) const;

    /**
     * @brief Deserializes all registered components of an entity
     *
     * @param file_path The file path of the serialized file
     * @param scene The scene where the entity belongs to
     * @param entity The entity that is serialized
     * @param j The json object describing this entity
     */
    void deserializeAllComponents(const std::string& file_path,
                                  const atcg::ref_ptr<Scene>& scene,
                                  Entity entity,
                                  nlohmann::json& j) const;

    /**
     * @brief Draws all registered components of an entity in the SceneHierarchyPanel
     *
     * @param scene The scene where the entity belongs to
     * @param entity The entity to display
     */
    void drawAllComponents(const atcg::ref_ptr<Scene>& scene, Entity entity) const;

    /**
     * @brief Display components in the "Add Component" Drop down menu
     *
     * @param scene The scene where the entity belongs to
     * @param entity The entity to display
     */
    void displayAddAllComponents(const atcg::ref_ptr<Scene>& scene, Entity entity) const;

    /**
     * @brief This function is used by the RevisionSystem to create a copy of the entity and its components
     *
     * @param entity The entity to copy
     * @param components The output components stored by the entity as generic pointers
     */
    void storeAllComponents(Entity entity, std::unordered_map<entt::id_type, std::shared_ptr<void>>& components) const;

    /**
     * @brief Restores and entity with its components
     *
     * @param entity The entity
     * @param id The id of the Component type (entt::hash)
     * @param component The component as generic pointer
     */
    void restoreAddAllComponents(Entity entity, const entt::id_type id, const std::shared_ptr<void>& component) const;

    /**
     * @brief Render all components
     *
     * @param renderer The renderer
     * @param entity The entity
     * @param camera The camera
     * @param auxiliary Dictionary with auxiliary information
     */
    void renderAllComponents(RendererSystem* renderer,
                             Entity entity,
                             const atcg::ref_ptr<Camera>& camera,
                             atcg::Dictionary& auxiliary) const;

private:
};

namespace ComponentRegistry
{
/**
 * @brief Serializes all registered components of an entity
 *
 * @param file_path The file path of the serialized file
 * @param scene The scene where the entity belongs to
 * @param entity The entity that is serialized
 * @param j The json object describing this entity
 */
ATCG_INLINE void serializeAllComponents(const std::string& file_name,
                                        const atcg::ref_ptr<Scene>& scene,
                                        Entity entity,
                                        nlohmann::json& j)
{
    SystemRegistry::instance()->getSystem<ComponentRegistrySystem>()->serializeAllComponents(file_name,
                                                                                             scene,
                                                                                             entity,
                                                                                             j);
}

/**
 * @brief Deserializes all registered components of an entity
 *
 * @param file_path The file path of the serialized file
 * @param scene The scene where the entity belongs to
 * @param entity The entity that is serialized
 * @param j The json object describing this entity
 */
ATCG_INLINE void deserializeAllComponents(const std::string& file_name,
                                          const atcg::ref_ptr<Scene>& scene,
                                          Entity entity,
                                          nlohmann::json& j)
{
    SystemRegistry::instance()->getSystem<ComponentRegistrySystem>()->deserializeAllComponents(file_name,
                                                                                               scene,
                                                                                               entity,
                                                                                               j);
}

/**
 * @brief Draws all registered components of an entity in the SceneHierarchyPanel
 *
 * @param scene The scene where the entity belongs to
 * @param entity The entity to display
 */
ATCG_INLINE void drawAllComponents(const atcg::ref_ptr<Scene>& scene, Entity entity)
{
    SystemRegistry::instance()->getSystem<ComponentRegistrySystem>()->drawAllComponents(scene, entity);
}

/**
 * @brief Display components in the "Add Component" Drop down menu
 *
 * @param scene The scene where the entity belongs to
 * @param entity The entity to display
 */
ATCG_INLINE void displayAddAllComponents(const atcg::ref_ptr<Scene>& scene, Entity entity)
{
    SystemRegistry::instance()->getSystem<ComponentRegistrySystem>()->displayAddAllComponents(scene, entity);
}

/**
 * @brief This function is used by the RevisionSystem to create a copy of the entity and its components
 *
 * @param entity The entity to copy
 * @param components The output components stored by the entity as generic pointers
 */
ATCG_INLINE void storeAllComponents(Entity entity, std::unordered_map<entt::id_type, std::shared_ptr<void>>& components)
{
    SystemRegistry::instance()->getSystem<ComponentRegistrySystem>()->storeAllComponents(entity, components);
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
    SystemRegistry::instance()->getSystem<ComponentRegistrySystem>()->restoreAddAllComponents(entity, id, component);
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
    SystemRegistry::instance()->getSystem<ComponentRegistrySystem>()->renderAllComponents(renderer,
                                                                                          entity,
                                                                                          camera,
                                                                                          auxiliary);
}
};    // namespace ComponentRegistry

#define ATCG_REGISTER_COMPONENT_SERIALIZATION(ComponentType)                                                           \
    struct SerializationFactory_##ComponentType                                                                        \
    {                                                                                                                  \
        SerializationFactory_##ComponentType()                                                                         \
        {                                                                                                              \
            getSerializationEntries().emplace_back(&Serialization::serializeComponent<ComponentType>,                  \
                                                   &Serialization::deserializeComponent<ComponentType>);               \
        }                                                                                                              \
        static SerializationFactory_##ComponentType instance;                                                          \
    };                                                                                                                 \
    SerializationFactory_##ComponentType SerializationFactory_##ComponentType::instance

#define ATCG_REGISTER_COMPONENT_DRAW(ComponentType)                                                                    \
    struct DrawFactory_##ComponentType                                                                                 \
    {                                                                                                                  \
        DrawFactory_##ComponentType()                                                                                  \
        {                                                                                                              \
            getDrawEntries().emplace_back(&GUI::drawComponent<ComponentType>,                                          \
                                          &GUI::displayAddComponentEntry<ComponentType>);                              \
        }                                                                                                              \
        static DrawFactory_##ComponentType instance;                                                                   \
    };                                                                                                                 \
    DrawFactory_##ComponentType DrawFactory_##ComponentType::instance

#define ATCG_REGISTER_COMPONENT_STORE(ComponentType)                                                                   \
    struct StoreFactory_##ComponentType                                                                                \
    {                                                                                                                  \
        StoreFactory_##ComponentType()                                                                                 \
        {                                                                                                              \
            getStoreEntries().emplace_back(&storeComponent<ComponentType>, &restoreComponent<ComponentType>);          \
        }                                                                                                              \
        static StoreFactory_##ComponentType instance;                                                                  \
    };                                                                                                                 \
    StoreFactory_##ComponentType StoreFactory_##ComponentType::instance

#define ATCG_REGISTER_COMPONENT_RENDER(ComponentType)                                                                  \
    struct RenderFactory_##ComponentType                                                                               \
    {                                                                                                                  \
        RenderFactory_##ComponentType() { getRenderEntries().emplace_back(&renderComponent<ComponentType>); }          \
        static RenderFactory_##ComponentType instance;                                                                 \
    };                                                                                                                 \
    RenderFactory_##ComponentType RenderFactory_##ComponentType::instance


#define ATCG_REGISTER_COMPONENT(ComponentType)                                                                         \
    ATCG_REGISTER_COMPONENT_SERIALIZATION(ComponentType);                                                              \
    ATCG_REGISTER_COMPONENT_DRAW(ComponentType);                                                                       \
    ATCG_REGISTER_COMPONENT_STORE(ComponentType);                                                                      \
    ATCG_REGISTER_COMPONENT_RENDER(ComponentType)

}    // namespace atcg