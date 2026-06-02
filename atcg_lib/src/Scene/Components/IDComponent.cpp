#include <Scene/Components/IDComponent.h>
#include <Scene/ComponentRegistry.h>

#define ID_KEY "ID"

namespace atcg
{

namespace Serialization
{

void ComponentSerializer<IDComponent>::serialize_component(const std::string& file_path,
                                                           const atcg::ref_ptr<Scene>& scene,
                                                           Entity entity,
                                                           IDComponent& component,
                                                           nlohmann::json& j) const
{
    j[ID_KEY] = (uint64_t)entity.getComponent<IDComponent>().ID();
}


void ComponentSerializer<IDComponent>::deserialize_component(const std::string& file_path,
                                                             const atcg::ref_ptr<Scene>& scene,
                                                             Entity entity,
                                                             nlohmann::json& j) const
{
    if(!j.contains(ID_KEY))
    {
        return;
    }

    entity.addOrReplaceComponent<IDComponent>((uint64_t)j[ID_KEY]);
}
}    // namespace Serialization

ATCG_REGISTER_COMPONENT_SERIALIZATION(IDComponent);
}    // namespace atcg