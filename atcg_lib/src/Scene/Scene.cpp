#include <Scene/Scene.h>

#include <Core/Assert.h>
#include <Scene/RevisionStack.h>
#include <Scene/Entity.h>
#include <Scene/Components.h>

#include <Renderer/RenderGraph.h>
#include <Renderer/RenderPasses/ForwardPass.h>
#include <Renderer/RenderPasses/SkyboxPass.h>
#include <Renderer/RenderPasses/ShadowPass.h>

namespace atcg
{

class Scene::Impl
{
public:
    Impl();
    ~Impl() = default;

    std::unordered_map<UUID, entt::entity> _entities;
    std::unordered_map<std::string, std::vector<entt::entity>> _entites_by_name;

    atcg::ref_ptr<atcg::Camera> _camera = nullptr;

    atcg::ref_ptr<atcg::RenderGraph> _render_graph;

    atcg::ref_ptr<Skybox> skybox;
    bool has_skybox = false;
};

Scene::Impl::Impl()
{
    // Skybox
    skybox = atcg::make_ref<Skybox>();

    _render_graph = atcg::make_ref<atcg::RenderGraph>();

    auto skybox_handle = _render_graph->addRenderPass(atcg::make_ref<SkyboxPass>(skybox));
    auto shadow_handle = _render_graph->addRenderPass(atcg::make_ref<ShadowPass>());
    auto output_handle = _render_graph->addRenderPass(atcg::make_ref<ForwardPass>(skybox));

    _render_graph->addDependency(skybox_handle, "framebuffer", output_handle, "framebuffer");
    _render_graph->addDependency(shadow_handle, "point_light_depth_maps", output_handle, "point_light_depth_maps");

    atcg::Dictionary context;    // TODO
    _render_graph->compile(context);
}

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

void Scene::draw(Dictionary& context)
{
    if(!context.contains("camera") && impl->_camera)
    {
        context.setValue("camera", impl->_camera);
    }

    if(!context.contains("camera"))
    {
        ATCG_WARN("Scene render was issued without valid camera");
        return;
    }

    context.setValue("scene", this);
    context.setValue("has_skybox", impl->has_skybox);

    impl->_render_graph->execute(context);
}

void Scene::draw(const atcg::ref_ptr<Camera>& camera)
{
    Dictionary context;
    context.setValue("camera", camera);
    draw(context);
}

void Scene::setSkybox(const atcg::ref_ptr<Image>& skybox)
{
    setSkybox(atcg::Texture2D::create(skybox));
}

void Scene::setSkybox(const atcg::ref_ptr<Texture2D>& skybox)
{
    impl->has_skybox = true;
    impl->skybox->setSkyboxTexture(skybox);
}

bool Scene::hasSkybox() const
{
    return impl->has_skybox;
}

void Scene::removeSkybox()
{
    impl->has_skybox = false;
}

atcg::ref_ptr<Texture2D> Scene::getSkyboxTexture() const
{
    return impl->skybox->getSkyboxTexture();
}

atcg::ref_ptr<TextureCube> Scene::getSkyboxCubemap() const
{
    return impl->skybox->getSkyboxCubeMap();
}

atcg::ref_ptr<Skybox> Scene::getSkybox() const
{
    return impl->skybox;
}

void Scene::setRenderGraph(const atcg::ref_ptr<RenderGraph>& graph)
{
    impl->_render_graph = graph;
}

atcg::ref_ptr<RenderGraph> Scene::getRenderGraph() const
{
    return impl->_render_graph;
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