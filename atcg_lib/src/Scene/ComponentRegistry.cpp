#include <Scene/ComponentRegistry.h>    // !

namespace atcg
{
namespace ComponentRegistry
{
void registerComponent(Registry* registry, std::string_view type, ComponentFunctions functions)
{
    registry->registerType(type, std::move(functions));
}

void serializeAllComponents(Registry* registry,
                            const std::string& file_path,
                            const atcg::ref_ptr<Scene>& scene,
                            Entity entity,
                            nlohmann::json& j)
{
    const auto& entries = registry->getEntries();
    for(const auto& [type, entry]: entries)
    {
        entry.desc.serialize(file_path, scene, entity, j);
    }
}


void deserializeAllComponents(Registry* registry,
                              const std::string& file_path,
                              const atcg::ref_ptr<Scene>& scene,
                              Entity entity,
                              nlohmann::json& j)
{
    const auto& entries = registry->getEntries();
    for(const auto& [type, entry]: entries)
    {
        entry.desc.deserialize(file_path, scene, entity, j);
    }
}

void drawAllComponents(Registry* registry, const atcg::ref_ptr<Scene>& scene, Entity entity)
{
    const auto& entries = registry->getEntries();
    for(const auto& [type, entry]: entries)
    {
        entry.desc.draw(scene, entity);
    }
}

void displayAddAllComponents(Registry* registry, const atcg::ref_ptr<Scene>& scene, Entity entity)
{
    const auto& entries = registry->getEntries();
    for(const auto& [type, entry]: entries)
    {
        entry.desc.display_add(scene, entity);
    }
}

void storeAllComponents(Registry* registry,
                        Entity entity,
                        std::unordered_map<entt::id_type, std::shared_ptr<void>>& components)
{
    const auto& entries = registry->getEntries();
    for(const auto& [type, entry]: entries)
    {
        entry.desc.store(entity, components);
    }
}

void restoreAddAllComponents(Registry* registry,
                             Entity entity,
                             const entt::id_type id,
                             const std::shared_ptr<void>& component)
{
    const auto& entries = registry->getEntries();
    for(const auto& [type, entry]: entries)
    {
        entry.desc.restore(entity, id, component);
    }
}

void renderAllComponents(Registry* registry,
                         RendererSystem* renderer,
                         Entity entity,
                         const atcg::ref_ptr<Camera>& camera,
                         atcg::Dictionary& auxiliary)
{
    const auto& entries = registry->getEntries();
    for(const auto& [type, entry]: entries)
    {
        entry.desc.render(renderer, entity, camera, auxiliary);
    }
}
}    // namespace ComponentRegistry
}    // namespace atcg