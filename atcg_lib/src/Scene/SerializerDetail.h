#pragma once

#include <fstream>

namespace atcg
{

namespace detail
{
template<typename ComponentSerializerT, typename Component>
void serializeComponent(const std::string& file_name,
                        const atcg::ref_ptr<ComponentSerializerT>& serializer,
                        Entity entity,
                        nlohmann::json& j)
{
    if(entity.hasComponent<Component>())
    {
        Component& component = entity.getComponent<Component>();
        serializer->template serialize_component<Component>(file_name, entity, component, j);
    }
}

template<typename ComponentSerializerT, typename Component>
void deserializeComponent(const std::string& file_name,
                          const atcg::ref_ptr<ComponentSerializerT>& serializer,
                          Entity entity,
                          nlohmann::json& j)
{
    serializer->template deserialize_component<Component>(file_name, entity, j);
}
}    // namespace detail

template<typename ComponentSerializerT>
template<typename... Components>
nlohmann::json Serializer<ComponentSerializerT>::serializeEntity(const std::string& file_name, Entity entity)
{
    auto entity_object = nlohmann::json::object();

    (detail::serializeComponent<ComponentSerializerT, Components>(file_name,
                                                                  _component_serializer,
                                                                  entity,
                                                                  entity_object),
     ...);

    return entity_object;
}

template<typename ComponentSerializerT>
template<typename... Components>
void Serializer<ComponentSerializerT>::deserializeEntity(const std::string& file_name,
                                                         Entity entity,
                                                         nlohmann::json& entity_object)
{
    (detail::deserializeComponent<ComponentSerializerT, Components>(file_name,
                                                                    _component_serializer,
                                                                    entity,
                                                                    entity_object),
     ...);
}

template<typename ComponentSerializerT>
Serializer<ComponentSerializerT>::Serializer(const atcg::ref_ptr<Scene>& scene)
    : _scene(scene),
      _component_serializer(atcg::make_ref<ComponentSerializerT>(scene))
{
}

template<typename ComponentSerializerT>
template<typename... CustomComponents>
ATCG_INLINE void Serializer<ComponentSerializerT>::serialize(const std::string& file_path)
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
                                             PointLightComponent,
                                             CustomComponents...>(file_path, entity);

        entity_array.push_back(entity_object);
    }

    j["Entities"] = entity_array;

    std::ofstream o(file_path);
    o << std::setw(4) << j << std::endl;
}

template<typename ComponentSerializerT>
template<typename... CustomComponents>
ATCG_INLINE void Serializer<ComponentSerializerT>::deserialize(const std::string& file_path)
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
                          PointLightComponent,
                          CustomComponents...>(file_path, entity, entity_object);
    }
}
}    // namespace atcg