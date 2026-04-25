#include <Scene/Components/NameComponent.h>
#include <Scene/ComponentRegistry.h>

#define NAME_KEY "Name"

namespace atcg
{

namespace Serialization
{
void ComponentSerializer<NameComponent>::serialize_component(const std::string& file_path,
                                                             const atcg::ref_ptr<Scene>& scene,
                                                             Entity entity,
                                                             NameComponent& component,
                                                             nlohmann::json& j) const
{
    j[NAME_KEY] = entity.getComponent<NameComponent>().name();
}

void ComponentSerializer<NameComponent>::deserialize_component(const std::string& file_path,
                                                               const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               nlohmann::json& j) const
{
    if(!j.contains(NAME_KEY))
    {
        return;
    }

    entity.addOrReplaceComponent<NameComponent>(j[NAME_KEY]);
}

}    // namespace Serialization

ATCG_REGISTER_COMPONENT_SERIALIZATION(NameComponent);
}    // namespace atcg