#include <Scene/Scene.h>

#include <Core/Assert.h>
#include <Scene/RevisionStack.h>
#include <Scene/Entity.h>
#include <Scene/Components.h>

namespace atcg
{

class Scene::Impl
{
public:
    Impl()  = default;
    ~Impl() = default;

    std::unordered_map<UUID, entt::entity> _entities;
    std::unordered_map<std::string, std::vector<entt::entity>> _entites_by_name;

    atcg::ref_ptr<atcg::Camera> _camera = nullptr;
};

Scene::Scene()
{
    impl = std::make_unique<Impl>();
}

Scene::~Scene() {}

Entity Scene::createEntity(const std::string& name)
{
    return createEntity((entt::entity)0, UUID(), name);
}

Entity Scene::createEntity(const entt::entity handle, UUID uuid, const std::string& name)
{
    Entity entity(_registry.create(handle), this);
    IDComponent id = entity.addComponent<IDComponent>(uuid);
    entity.addComponent<NameComponent>(name);

    impl->_entities.insert(std::make_pair(id.ID(), (entt::entity)entity.entity_handle()));

    auto& entities = impl->_entites_by_name[name];
    entities.push_back((entt::entity)entity.entity_handle());

    return entity;
}

Entity Scene::getEntityByID(UUID id) const
{
    if(impl->_entities.find(id) == impl->_entities.end()) return Entity();
    return Entity(impl->_entities.find(id)->second, (Scene*)this);
}

std::vector<Entity> Scene::getEntitiesByName(const std::string& name)
{
    const std::vector<entt::entity>& entities = impl->_entites_by_name[name];
    std::vector<Entity> res_entities;
    res_entities.reserve(entities.size());
    for(auto& e: entities)
    {
        res_entities.emplace_back(e, this);
    }
    return res_entities;
}

void Scene::removeEntity(UUID id)
{
    auto it_entity = impl->_entities.find(id);
    if(it_entity == impl->_entities.end()) return;

    auto entity_handle = it_entity->second;
    Entity entity(entity_handle, this);
    auto& name = entity.getComponent<atcg::NameComponent>();

    impl->_entities.erase(id);
    auto& entities_with_name = impl->_entites_by_name[name.name()];

    for(auto it = entities_with_name.begin(); it != entities_with_name.end(); ++it)
    {
        Entity other_entity(*it, this);
        auto& other_id = other_entity.getComponent<atcg::IDComponent>();
        if(other_id.ID() == id)
        {
            entities_with_name.erase(it);
            break;
        }
    }

    _registry.destroy(entity._entity_handle);
}

void Scene::removeEntity(Entity entity)
{
    auto& id = entity.getComponent<atcg::IDComponent>();
    removeEntity(id.ID());
}

void Scene::removeAllEntites()
{
    _registry.clear();
    impl->_entites_by_name.clear();
    impl->_entities.clear();

    atcg::RevisionStack::clearChache();
}

void Scene::setCamera(const atcg::ref_ptr<atcg::Camera>& camera)
{
    impl->_camera = camera;
}

atcg::ref_ptr<atcg::Camera> Scene::getCamera() const
{
    return impl->_camera;
}

void Scene::removeCamera()
{
    setCamera(nullptr);
}

void Scene::_updateEntityID(atcg::Entity entity, const UUID old_id, const UUID new_id)
{
    impl->_entities.erase(old_id);
    impl->_entities.insert(std::make_pair(new_id, (entt::entity)entity.entity_handle()));
}

void Scene::_updateEntityName(atcg::Entity entity, const std::string& old_name, const std::string& new_name)
{
    auto& old_list = impl->_entites_by_name[old_name];
    for(auto it = old_list.begin(); it != old_list.end(); ++it)
    {
        if(*it == (entt::entity)entity.entity_handle())
        {
            old_list.erase(it);
            break;
        }
    }

    auto& entities = impl->_entites_by_name[new_name];
    entities.push_back((entt::entity)entity.entity_handle());
}


}    // namespace atcg