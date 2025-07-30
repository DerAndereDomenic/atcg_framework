#include <Scene/ComponentRegistry.h>

namespace atcg
{

void ComponentRegistrySystem::serializeAllComponents(const std::string& file_name,
                                                     const atcg::ref_ptr<Scene>& scene,
                                                     Entity entity,
                                                     nlohmann::json& j) const
{
    for(const auto& entry: getRegistryEntries())
    {
        if(entry.serialize) entry.serialize(file_name, scene, entity, j);
    }
}

void ComponentRegistrySystem::deserializeAllComponents(const std::string& file_name,
                                                       const atcg::ref_ptr<Scene>& scene,
                                                       Entity entity,
                                                       nlohmann::json& j) const
{
    for(const auto& entry: getRegistryEntries())
    {
        if(entry.deserialize) entry.deserialize(file_name, scene, entity, j);
    }
}

void ComponentRegistrySystem::drawAllComponents(const atcg::ref_ptr<Scene>& scene, Entity entity) const
{
    for(const auto& entry: getRegistryEntries())
    {
        if(entry.draw) entry.draw(scene, entity);
    }
}

void ComponentRegistrySystem::displayAddAllComponents(const atcg::ref_ptr<Scene>& scene, Entity entity) const
{
    for(const auto& entry: getRegistryEntries())
    {
        if(entry.display_add) entry.display_add(scene, entity);
    }
}

void ComponentRegistrySystem::storeAllComponents(
    Entity entity,
    std::unordered_map<entt::id_type, std::shared_ptr<void>>& components) const
{
    for(const auto& entry: getRegistryEntries())
    {
        if(entry.store) entry.store(entity, components);
    }
}

void ComponentRegistrySystem::restoreAddAllComponents(Entity entity,
                                                      const entt::id_type id,
                                                      const std::shared_ptr<void>& component) const
{
    for(const auto& entry: getRegistryEntries())
    {
        if(entry.restore) entry.restore(entity, id, component);
    }
}
}    // namespace atcg