#pragma once

#include <Scene/Scene.h>
#include <Scene/Components.h>

#include <json.hpp>

namespace atcg
{
/**
 * @brief A class that handles the serialization of components
 *
 * To add custom serialization code, create a class that inherits from this class and add the serialization code for the
 * custom component. To get the default behavior, override the serialize_component and deserialize_component function
 * that calls the super class. Add this class as template argument to the atcg::Serializer.
 *
 * @code{.cpp}
 * class MyComponentSerializer : ComponentSerializer
 * {
 * public:
 *      MyComponentSerializer(atcg::ref_ptr<Scene>& scene) :ComponentSerializer(scene) {}
 *
 *      template<typename T>
 *      void serialize_component(const std::string& file_path, Entity entity, T& component, nlohmann::json& j)
 *      {
 *          ComponentSerializer::serialize_component<T>(file_path, entity, component, j);
 *      }
 *
 *      template<>
 *      void serialize_component<MyCustomComponent>(const std::string& file_path,
 *                                             Entity entity, T& component,
 *                                             nlohmann::json& j)
 *      {
 *          // Serialize custom component
 *      }
 *
 *      template<typename T>
 *      void deserialize_component(const std::string& file_path, Entity entity, nlohmann::json& j)
 *      {
 *          ComponentSerializer::deserialize_component<T>(file_path, entity, j);
 *      }
 *
 *      template<>
 *      void deserialize_component<MyCustomComponent>(const std::string& file_path, Entity entity, nlohmann::json& j)
 *      {
 *          // Serialize custom component
 *      }
 * };
 * @endcode
 */
class ComponentSerializer
{
public:
    /**
     * @brief Constructor
     *
     * @param scene The scene
     */
    ComponentSerializer(const atcg::ref_ptr<Scene>& scene) : _scene(scene) {}

    /**
     * @brief Serialize a component
     *
     * @tparam T The component type
     *
     * @param file_path The file path of the serialized file
     * @param entity The serialized entity
     * @param component The component to serialize
     * @param j The json representing the serialized entity
     */
    template<typename T>
    void serialize_component(const std::string& file_path, Entity entity, T& component, nlohmann::json& j);

    /**
     * @brief Deserialize a component
     *
     * @tparam T The component type
     *
     * @param file_path The file path of the serialized file
     * @param entity The deserialized entity
     * @param j The json representing the serialized entity
     */
    template<typename T>
    void deserialize_component(const std::string& file_path, Entity entity, nlohmann::json& j);

protected:
    void serializeBuffer(const std::string& file_name, const char* data, const uint32_t byte_size);
    std::vector<uint8_t> deserializeBuffer(const std::string& file_name);
    void serializeMaterial(nlohmann::json& out, Entity entity, const Material& material, const std::string& file_path);
    Material deserialize_material(const nlohmann::json& material_node);
    void serializeTexture(const atcg::ref_ptr<Texture2D>& texture, std::string& path, float gamma = 1.0f);
    nlohmann::json serializeLayout(const atcg::BufferLayout& layout);
    atcg::BufferLayout deserializeLayout(nlohmann::json& layout_node);
    atcg::ref_ptr<Scene> _scene;
};
}    // namespace atcg