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

Entity Scene::createEntity()
{
    Entity entity(_registry.create(), this);
    IDComponent id = entity.addComponent<IDComponent>();

    impl->_entities.insert(std::make_pair(id.ID, entity));
    return entity;
}

Entity Scene::getEntityByID(UUID id) const
{
    return impl->_entities.find(id)->second;
}
}    // namespace atcg