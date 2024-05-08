#pragma once

#include <Core/Memory.h>
#include <Scene/Entity.h>

#include <json.hpp>

namespace atcg
{

/**
 * @brief A class that handles scene serialization
 *
 * @tparam ComponentSerializerT The class that handles serialization of individual components
 */
template<typename ComponentSerializerT = ComponentSerializer>
class Serializer
{
public:
    /**
     * @brief Constructor.
     * The given scene is either serialized or the deserialized contents is added to the given scene.
     *
     * @param scene The scene.
     */
    Serializer(const atcg::ref_ptr<Scene>& scene);

    /**
     * @brief Serialize the scene.
     *
     * @tparam CustomComponents... Optional custom components to serialize.
     * 
     * To use custom components, a custom serializer has to be supplied.
     * 
     * @param file_path The file path
     */
    template<typename... CustomComponents>
    void serialize(const std::string& file_path);

    /**
     * @brief Deserialize the scene.
     *
     * @tparam CustomComponents... Optional custom components to serialize.
     * 
     * To use custom components, a custom serializer has to be supplied.
     * 
     * @param file_path The file path
     */
    template<typename... CustomComponents>
    void deserialize(const std::string& file_path);

private:
    template<typename... Components>
    nlohmann::json serializeEntity(const std::string& file_path, Entity entity);
    template<typename... Components>
    void deserializeEntity(const std::string& file_path, Entity entity, nlohmann::json& entity_object);

    atcg::ref_ptr<Scene> _scene;
    atcg::ref_ptr<ComponentSerializerT> _component_serializer;
};
}    // namespace atcg

#include "../../src/Scene/SerializerDetail.h"