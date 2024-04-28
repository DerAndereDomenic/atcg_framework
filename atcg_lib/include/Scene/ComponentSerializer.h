#pragma once

#include <Scene/Scene.h>
#include <Scene/Components.h>

#include <json.hpp>

namespace atcg
{
class ComponentSerializer
{
public:
    ComponentSerializer(const atcg::ref_ptr<Scene>& scene) : _scene(scene) {}

    template<typename T>
    void serialize_component(const std::string& file_path, Entity entity, T& component, nlohmann::json& j);

protected:
    void serializeBuffer(const std::string& file_name, const char* data, const uint32_t byte_size);
    void serializeMaterial(nlohmann::json& out, Entity entity, const Material& material, const std::string& file_path);
    void serializeTexture(const atcg::ref_ptr<Texture2D>& texture, std::string& path, float gamma = 1.0f);
    atcg::ref_ptr<Scene> _scene;
};
}    // namespace atcg