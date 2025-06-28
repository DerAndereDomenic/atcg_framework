#pragma once

#include <fstream>

#include <Scene/ComponentSerializer.h>

namespace atcg
{

namespace Serialization
{

namespace detail
{
template<typename Component>
ATCG_INLINE void
serializeComponent(const std::string& file_name, const atcg::ref_ptr<Scene>& scene, Entity entity, nlohmann::json& j)
{
    if(entity.hasComponent<Component>())
    {
        Component& component = entity.getComponent<Component>();
        ComponentSerializer<Component>().serialize_component(file_name, scene, entity, component, j);
    }
}

template<typename Component>
ATCG_INLINE void
deserializeComponent(const std::string& file_name, const atcg::ref_ptr<Scene>& scene, Entity entity, nlohmann::json& j)
{
    ComponentSerializer<Component>().deserialize_component(file_name, scene, entity, j);
}
}    // namespace detail

template<typename... Components>
ATCG_INLINE nlohmann::json SceneSerializer::serializeEntity(const std::string& file_name, Entity entity)
{
    auto entity_object = nlohmann::json::object();

    (detail::serializeComponent<Components>(file_name, _scene, entity, entity_object), ...);

    return entity_object;
}

template<typename... Components>
ATCG_INLINE void
SceneSerializer::deserializeEntity(const std::string& file_name, Entity entity, nlohmann::json& entity_object)
{
    (detail::deserializeComponent<Components>(file_name, _scene, entity, entity_object), ...);
}

ATCG_INLINE SceneSerializer::SceneSerializer(const atcg::ref_ptr<Scene>& scene) : _scene(scene) {}

template<typename... CustomComponents>
ATCG_INLINE void SceneSerializer::serialize(const std::string& file_path)
{
    nlohmann::json j;

    j["Scene"] = "Untitled";

    auto entity_array = nlohmann::json::array();

    auto entity_view = _scene->getAllEntitiesWith<IDComponent>();
    for(auto e: entity_view)
    {
        Entity entity(e, _scene.get());

        auto entity_object = serializeEntity<IDComponent,
                                             NameComponent,
                                             TransformComponent,
                                             CameraComponent,
                                             GeometryComponent,
                                             MeshRenderComponent,
                                             PointRenderComponent,
                                             PointSphereRenderComponent,
                                             EdgeRenderComponent,
                                             EdgeCylinderRenderComponent,
                                             InstanceRenderComponent,
                                             PointLightComponent,
                                             ScriptComponent,
                                             CustomComponents...>(file_path, entity);

        entity_array.push_back(entity_object);
    }

    j["Entities"] = entity_array;

    std::ofstream o(file_path);
    o << std::setw(4) << j << std::endl;
}

template<typename... CustomComponents>
ATCG_INLINE void SceneSerializer::deserialize(const std::string& file_path)
{
    std::ifstream i(file_path);
    nlohmann::json j;
    i >> j;

    if(!j.contains("Entities"))
    {
        return;
    }

    auto entities = j["Entities"];

    for(auto entity_object: entities)
    {
        if(!entity_object.contains("Name")) continue;
        Entity entity = _scene->createEntity(entity_object["Name"]);

        deserializeEntity<IDComponent,
                          NameComponent,
                          TransformComponent,
                          CameraComponent,
                          GeometryComponent,
                          MeshRenderComponent,
                          PointRenderComponent,
                          PointSphereRenderComponent,
                          EdgeRenderComponent,
                          EdgeCylinderRenderComponent,
                          InstanceRenderComponent,
                          PointLightComponent,
                          ScriptComponent,
                          CustomComponents...>(file_path, entity, entity_object);
    }
}
}    // namespace Serialization
}    // namespace atcg