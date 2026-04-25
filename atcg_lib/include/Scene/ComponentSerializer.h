#pragma once

#include <Scene/Scene.h>

#include <json.hpp>

namespace atcg
{
namespace Serialization
{
/**
 * @brief A class that handles the serialization of components
 *
 * @tparam T The component to serialize
 *
 * To add custom serialization code, create a template specialization that implements the serialize_component and
 * deserialize_component function.
 *
 * @code{.cpp}
 * template<>
 * struct atcg::Serialization::ComponentSerializer<CustomComponent>
 * {
 *     void serialize_component(const std::string& file_path,
 *                              const atcg::ref_ptr<Scene>& scene,
 *                              Entity entity,
 *                              CustomComponent& component,
 *                              nlohmann::json& j) const
 *     {
 *         j["CustomComponent"]["Content"] = component.content;
 *     }

 *     void deserialize_component(const std::string& file_path,
 *                              const atcg::ref_ptr<Scene>& scene,
 *                              Entity entity,
 *                              nlohmann::json& j) const
 *     {
 *         if(!j.contains("CustomComponent"))
 *         {
 *             return;
 *         }

 *         auto& component   = entity.addComponent<CustomComponent>();
 *         component.content = j["CustomComponent"]["Content"];
 *     }
 * };
 * @endcode
 */
template<typename T>
struct ComponentSerializer
{
    /**
     * @brief Serialize a component
     *
     * @param file_path The file_path of the serialized scene. This can be used to store additional buffers in the same
     * directory
     * @param scene The scene to which the entity holding the component belongs to
     * @param entity The entity that holds the component
     * @param component The component to serialize
     * @param j The json object
     */
    void serialize_component(const std::string& file_path,
                             const atcg::ref_ptr<Scene>& scene,
                             Entity entity,
                             T& component,
                             nlohmann::json& j) const
    {
    }

    /**
     * @brief Deserialize a component
     *
     * @param file_path The file_path of the serialized scene. This can be used to store additional buffers in the same
     * directory
     * @param scene The scene to which the entity holding the component belongs to
     * @param entity The entity that holds the component
     * @param j The json object
     */
    void deserialize_component(const std::string& file_path,
                               const atcg::ref_ptr<Scene>& scene,
                               Entity entity,
                               nlohmann::json& j) const
    {
    }
};

#define ATCG_DECLARE_COMPONENT_SERIALIZER(ComponentType)                                                               \
    template<>                                                                                                         \
    struct ComponentSerializer<ComponentType>                                                                          \
    {                                                                                                                  \
        void serialize_component(const std::string& file_path,                                                         \
                                 const atcg::ref_ptr<Scene>& scene,                                                    \
                                 Entity entity,                                                                        \
                                 ComponentType& component,                                                             \
                                 nlohmann::json& j) const;                                                             \
                                                                                                                       \
        void deserialize_component(const std::string& file_path,                                                       \
                                   const atcg::ref_ptr<Scene>& scene,                                                  \
                                   Entity entity,                                                                      \
                                   nlohmann::json& j) const;                                                           \
    }


template<typename ComponentType>
ATCG_INLINE void
serializeComponent(const std::string& file_name, const atcg::ref_ptr<Scene>& scene, Entity entity, nlohmann::json& j)
{
    if(entity.hasComponent<ComponentType>())
    {
        ComponentType& component = entity.getComponent<ComponentType>();
        ComponentSerializer<ComponentType>().serialize_component(file_name, scene, entity, component, j);
    }
}

template<typename ComponentType>
ATCG_INLINE void
deserializeComponent(const std::string& file_name, const atcg::ref_ptr<Scene>& scene, Entity entity, nlohmann::json& j)
{
    ComponentSerializer<ComponentType>().deserialize_component(file_name, scene, entity, j);
}
}    // namespace Serialization
}    // namespace atcg