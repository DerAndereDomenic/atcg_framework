#pragma once

#include <Core/API.h>
#include <Core/Memory.h>
#include <Scene/Entity.h>

#include <json.hpp>

namespace atcg
{

namespace Serialization
{

/**
 * @brief A class that handles scene serialization
 */
class ATCG_API SceneSerializer
{
public:
    /**
     * @brief Constructor.
     * The given scene is either serialized or the deserialized contents is added to the given scene.
     *
     * @param scene The scene.
     */
    SceneSerializer(const atcg::ref_ptr<Scene>& scene);

    /**
     * @brief Serialize the scene.
     *
     * @param file_path The file path
     */
    void serialize(const std::string& file_path);

    /**
     * @brief Deserialize the scene.
     *
     * @param file_path The file path
     */
    void deserialize(const std::string& file_path);

private:
    nlohmann::json serializeEntity(const std::string& file_path, Entity entity);

    void deserializeEntity(const std::string& file_path, Entity entity, nlohmann::json& entity_object);

    atcg::ref_ptr<Scene> _scene;
};
}    // namespace Serialization
}    // namespace atcg