#pragma once

#include <Core/SystemRegistry.h>
#include <Scene/Entity.h>
#include <Scene/ComponentSerializer.h>
#include <Scene/ComponentGUIHandler.h>

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

struct ComponentRegistryEntry
{
    serialize_fn serialize;
    deserialize_fn deserialize;
    draw_fn draw;
    display_add_fn display_add;
    store_fn store;
    restore_fn restore;
};

ATCG_INLINE std::vector<ComponentRegistryEntry>& getRegistryEntries()
{
    static std::vector<ComponentRegistryEntry> entries;
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
 * namespace atcg
 * {
 * ATCG_REGISTER_COMPONENT(CustomComponent);
 * }
 * @endcode
 */
class ComponentRegistrySystem
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
};    // namespace ComponentRegistry

#define ATCG_REGISTER_COMPONENT(ComponentType)                                                                         \
    struct RegistryFactory_##ComponentType                                                                             \
    {                                                                                                                  \
        RegistryFactory_##ComponentType()                                                                              \
        {                                                                                                              \
            getRegistryEntries().push_back({&Serialization::serializeComponent<ComponentType>,                         \
                                            &Serialization::deserializeComponent<ComponentType>,                       \
                                            &GUI::drawComponent<ComponentType>,                                        \
                                            &GUI::displayAddComponentEntry<ComponentType>,                             \
                                            &storeComponent<ComponentType>,                                            \
                                            &restoreComponent<ComponentType>});                                        \
        }                                                                                                              \
        static RegistryFactory_##ComponentType instance;                                                               \
    };                                                                                                                 \
    RegistryFactory_##ComponentType RegistryFactory_##ComponentType::instance

#define ATCG_REGISTER_COMPONENT_SERIALIZATION_ONLY(ComponentType)                                                      \
    struct RegistryFactory_##ComponentType                                                                             \
    {                                                                                                                  \
        RegistryFactory_##ComponentType()                                                                              \
        {                                                                                                              \
            getRegistryEntries().push_back({&Serialization::serializeComponent<ComponentType>,                         \
                                            &Serialization::deserializeComponent<ComponentType>,                       \
                                            {},                                                                        \
                                            {},                                                                        \
                                            {},                                                                        \
                                            {}});                                                                      \
        }                                                                                                              \
        static RegistryFactory_##ComponentType instance;                                                               \
    };                                                                                                                 \
    RegistryFactory_##ComponentType RegistryFactory_##ComponentType::instance

}    // namespace atcg