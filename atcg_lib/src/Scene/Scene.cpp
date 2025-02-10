#include <Scene/Scene.h>

#include <Core/Assert.h>
#include <Scene/Entity.h>
#include <Scene/Components.h>

namespace atcg
{

class Scene::Impl
{
public:
    Impl()  = default;
    ~Impl() = default;

    std::unordered_map<UUID, Entity> _entities;
    std::unordered_map<std::string, std::vector<Entity>> _entites_by_name;
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

    impl->_entities.insert(std::make_pair(id.ID, entity));

    auto& entities = impl->_entites_by_name[name];
    entities.push_back(entity);

    return entity;
}

Entity Scene::getEntityByID(UUID id) const
{
    if(impl->_entities.find(id) == impl->_entities.end()) return Entity();
    return impl->_entities.find(id)->second;
}

std::vector<Entity> Scene::getEntitiesByName(const std::string& name)
{
    return impl->_entites_by_name[name];
}

void Scene::removeEntity(UUID id)
{
    auto it_entity = impl->_entities.find(id);
    if(it_entity == impl->_entities.end()) return;

    auto entity = it_entity->second;
    auto& name  = entity.getComponent<atcg::NameComponent>();

    impl->_entities.erase(id);
    auto& entities_with_name = impl->_entites_by_name[name.name];

    for(auto it = entities_with_name.begin(); it != entities_with_name.end(); ++it)
    {
        auto& other_id = it->getComponent<atcg::IDComponent>();
        if(other_id.ID == id)
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
    removeEntity(id.ID);
}

void Scene::removeAllEntites()
{
    _registry.clear();
    impl->_entites_by_name.clear();
    impl->_entities.clear();
}

}    // namespace atcg