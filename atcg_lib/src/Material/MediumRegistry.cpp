#include <Material/MediumRegistry.h>

namespace atcg
{
namespace MediumRegistry
{
void registerMedium(Registry* registry, std::string_view type, MediumFunctions functions)
{
    registry->registerType(type, std::move(functions));
}

atcg::ref_ptr<Medium> createMedium(Registry* registry, const std::string& type, const Dictionary& dict)
{
    const MediumFunctions* desc = registry->find(type);
    return desc->builder(dict);
}

bool renderMediumGUI(Registry* registry, const atcg::ref_ptr<Medium>& medium, const std::string& key, bool& deactivated)
{
    const MediumFunctions* desc = registry->find(medium->getMediumType());
    return desc->gui_function(medium, key, deactivated);
}

const std::vector<std::string>& getRegisteredMediumTypes(Registry* registry)
{
    return registry->getRegisteredTypes();
}

void serializeMedium(Registry* registry, const atcg::ref_ptr<Medium>& medium, const std::filesystem::path& path)
{
    const MediumFunctions* desc = registry->find(medium->getMediumType());
    desc->serializer_function(medium, path);
}

atcg::ref_ptr<Medium> deserializeMedium(Registry* registry,
                                        std::string_view medium_type,
                                        const std::filesystem::path& path,
                                        const nlohmann::json& medium_node)
{
    const MediumFunctions* desc = registry->find(medium_type);
    return desc->deserializer_function(path, medium_node);
}
}    // namespace MediumRegistry
}    // namespace atcg