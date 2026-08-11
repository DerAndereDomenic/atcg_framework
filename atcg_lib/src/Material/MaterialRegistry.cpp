#include <Material/MaterialRegistry.h>

namespace atcg
{
namespace MaterialRegistry
{
void registerMaterial(Registry* registry, std::string_view type, MaterialFunctions functions)
{
    registry->registerType(type, std::move(functions));
}

atcg::ref_ptr<Material> createMaterial(Registry* registry, const std::string& type, const Dictionary& dict)
{
    const MaterialFunctions* desc = registry->find(type);
    return desc->builder(dict);
}

bool renderMaterialGUI(Registry* registry,
                       const atcg::ref_ptr<Material>& material,
                       const std::string& key,
                       bool& deactivated)
{
    const MaterialFunctions* desc = registry->find(material->getMaterialType());
    return desc->gui_function(material, key, deactivated);
}

const std::vector<std::string>& getRegisteredMaterialTypes(Registry* registry)
{
    return registry->getRegisteredTypes();
}

void serializeMaterial(Registry* registry, const atcg::ref_ptr<Material>& material, const std::filesystem::path& path)
{
    const MaterialFunctions* desc = registry->find(material->getMaterialType());
    desc->serializer_function(material, path);
}

atcg::ref_ptr<Material> deserializeMaterial(Registry* registry,
                                            std::string_view material_type,
                                            const std::filesystem::path& path,
                                            const nlohmann::json& material_node)
{
    const MaterialFunctions* desc = registry->find(material_type);
    return desc->deserializer_function(path, material_node);
}
}    // namespace MaterialRegistry
}    // namespace atcg