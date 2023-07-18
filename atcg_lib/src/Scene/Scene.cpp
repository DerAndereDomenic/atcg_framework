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
    return entity;
}

Entity Scene::getEntityByID(UUID id) const
{
    return impl->_entities.find(id)->second;
}

std::vector<Entity> Scene::getEntitiesByName(const std::string& name)
{
    std::vector<Entity> entities;
    auto& view = _registry.view<NameComponent>();
    for(auto e: view)
    {
        Entity entity(e, this);
        if(name == entity.getComponent<NameComponent>().name) { entities.push_back(entity); }
    }

    return entities;
}

}    // namespace atcg