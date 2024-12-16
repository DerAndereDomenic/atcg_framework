#include <Scene/Scene.h>

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
    Entity entity(_registry.create(), this);
    IDComponent id = entity.addComponent<IDComponent>();
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

}    // namespace atcg