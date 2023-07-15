#pragma once

#include <Core/Memory.h>
#include <Scene/Scene.h>

namespace atcg
{

/**
 * @brief A class that handles scene serialization
 */
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
    atcg::ref_ptr<Scene> _scene;
};
}    // namespace atcg