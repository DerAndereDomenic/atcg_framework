#pragma once

#include <Scene/Scene.h>
#include <Scene/Components.h>

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
        throw std::logic_error("No ComponentSerializer specialization available for this component type");
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
        throw std::logic_error("No ComponentSerializer specialization available for this component type");
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

ATCG_DECLARE_COMPONENT_SERIALIZER(IDComponent);
ATCG_DECLARE_COMPONENT_SERIALIZER(NameComponent);
ATCG_DECLARE_COMPONENT_SERIALIZER(TransformComponent);
ATCG_DECLARE_COMPONENT_SERIALIZER(CameraComponent);
ATCG_DECLARE_COMPONENT_SERIALIZER(GeometryComponent);
ATCG_DECLARE_COMPONENT_SERIALIZER(MeshRenderComponent);
ATCG_DECLARE_COMPONENT_SERIALIZER(PointRenderComponent);
ATCG_DECLARE_COMPONENT_SERIALIZER(PointSphereRenderComponent);
ATCG_DECLARE_COMPONENT_SERIALIZER(EdgeRenderComponent);
ATCG_DECLARE_COMPONENT_SERIALIZER(EdgeCylinderRenderComponent);
ATCG_DECLARE_COMPONENT_SERIALIZER(InstanceRenderComponent);
ATCG_DECLARE_COMPONENT_SERIALIZER(PointLightComponent);
ATCG_DECLARE_COMPONENT_SERIALIZER(ScriptComponent);

/**
 * @brief Serialize a buffer
 *
 * @param file_name The file name
 * @param data The buffer data
 * @param byte_size The buffer size in bytes
 */
void serializeBuffer(const std::string& file_name, const char* data, const uint32_t byte_size);

/**
 * @brief Deserialize a buffer
 *
 * @param file_name The file name
 *
 * @return The deserialized data
 */
std::vector<uint8_t> deserializeBuffer(const std::string& file_name);

/**
 * @brief Serialize a layout
 *
 * @param layout The buffer layout
 *
 * @return The json object representing the layout
 */
nlohmann::json serializeLayout(const atcg::BufferLayout& layout);

/**
 * @brief Deserialize a layout
 *
 * @param layout_node The json node containing the Layout data
 *
 * @return The BufferLayout
 */
atcg::BufferLayout deserializeLayout(nlohmann::json& layout_node);
}    // namespace Serialization
}    // namespace atcg