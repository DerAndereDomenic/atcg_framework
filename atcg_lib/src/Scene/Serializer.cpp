#include <Scene/Serializer.h>

#include <fstream>

#include <Scene/ComponentSerializer.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{

namespace Serialization
{

nlohmann::json SceneSerializer::serializeEntity(const std::string& file_name, Entity entity)
{
    auto entity_object = nlohmann::json::object();

    ComponentRegistry::serializeAllComponents(file_name, _scene, entity, entity_object);

    return entity_object;
}

void SceneSerializer::deserializeEntity(const std::string& file_name, Entity entity, nlohmann::json& entity_object)
{
    ComponentRegistry::deserializeAllComponents(file_name, _scene, entity, entity_object);
}

SceneSerializer::SceneSerializer(const atcg::ref_ptr<Scene>& scene) : _scene(scene) {}

void SceneSerializer::serialize(const std::string& file_path)
{
    nlohmann::json j;

    auto entity_array = nlohmann::json::array();

    auto entity_view = _scene->getAllEntitiesWith<IDComponent>();
    for(auto e: entity_view)
    {
        Entity entity(e, _scene.get());

        auto entity_object = serializeEntity(file_path, entity);

        entity_array.push_back(entity_object);
    }

    j["Entities"] = entity_array;
    j["Version"]  = "1.0";

    if(_scene->hasSkybox())
    {
        j["Skybox"] = (uint64_t)_scene->getSkyboxTexture()->handle;
    }

    std::ofstream o(file_path);
    o << std::setw(4) << j << std::endl;
}

void SceneSerializer::deserialize(const std::string& file_path)
{
    std::ifstream i(file_path);
    nlohmann::json j;
    i >> j;

    if(!j.contains("Entities"))
    {
        return;
    }

    auto entities = j["Entities"];

    if(j.contains("Skybox"))
    {
        AssetHandle skybox_handle = (AssetHandle)j["Skybox"];
        if(AssetManager::isAssetHandleValid(skybox_handle))
        {
            auto skybox_texture = AssetManager::getAsset<Texture2D>(skybox_handle);
            _scene->setSkybox(skybox_texture);
        }
    }

    for(auto entity_object: entities)
    {
        if(!entity_object.contains("Name")) continue;
        Entity entity = _scene->createEntity(entity_object["Name"]);

        deserializeEntity(file_path, entity, entity_object);
    }
}
}    // namespace Serialization
}    // namespace atcg