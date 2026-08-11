#pragma once

#include <Material/Material.h>
#include <DataStructure/Registry.h>

namespace atcg
{

namespace MaterialRegistry
{
using MaterialBuilder           = std::function<atcg::ref_ptr<Material>(const Dictionary&)>;
using MaterialGUIFunction       = std::function<bool(const atcg::ref_ptr<Material>&, const std::string&, bool&)>;
using MaterialSerializeFunction = std::function<void(const atcg::ref_ptr<Material>&, const std::filesystem::path&)>;
using MaterialDeserializeFunction =
    std::function<atcg::ref_ptr<Material>(const std::filesystem::path&, const nlohmann::json&)>;
struct MaterialFunctions
{
    MaterialBuilder builder;
    MaterialGUIFunction gui_function;
    MaterialSerializeFunction serializer_function;
    MaterialDeserializeFunction deserializer_function;
};

using Registry = atcg::Registry<MaterialFunctions>;

ATCG_API void registerMaterial(Registry* registry, std::string_view type, MaterialFunctions);

ATCG_API atcg::ref_ptr<Material> createMaterial(Registry* registry, const std::string& type, const Dictionary& dict);

ATCG_API bool renderMaterialGUI(Registry* registry,
                                const atcg::ref_ptr<Material>& material,
                                const std::string& key,
                                bool& deactivated);

ATCG_API const std::vector<std::string>& getRegisteredMaterialTypes(Registry* registry);

ATCG_API void
serializeMaterial(Registry* registry, const atcg::ref_ptr<Material>& material, const std::filesystem::path& path);

ATCG_API atcg::ref_ptr<Material> deserializeMaterial(Registry* registry,
                                                     std::string_view material_type,
                                                     const std::filesystem::path& path,
                                                     const nlohmann::json& material_node);

ATCG_INLINE Registry* getRegistry()
{
    Registry* registry = SystemRegistry::instance()->getSystem<MaterialRegistry::Registry>();
    ATCG_ASSERT(registry, "Material registry not found");
    return registry;
}

ATCG_INLINE void registerMaterial(std::string_view type, MaterialFunctions functions)
{
    registerMaterial(getRegistry(), type, std::move(functions));
}

ATCG_INLINE atcg::ref_ptr<Material> createMaterial(const std::string& type, const Dictionary& dict)
{
    return createMaterial(getRegistry(), type, dict);
}

ATCG_INLINE bool renderMaterialGUI(const atcg::ref_ptr<Material>& material, const std::string& key, bool& deactivated)
{
    return renderMaterialGUI(getRegistry(), material, key, deactivated);
}

ATCG_INLINE const std::vector<std::string>& getRegisteredMaterialTypes()
{
    return getRegisteredMaterialTypes(getRegistry());
}

ATCG_INLINE void serializeMaterial(const atcg::ref_ptr<Material>& material, const std::filesystem::path& path)
{
    serializeMaterial(getRegistry(), material, path);
}

ATCG_INLINE atcg::ref_ptr<Material> deserializeMaterial(std::string_view material_type,
                                                        const std::filesystem::path& path,
                                                        const nlohmann::json& material_node)
{
    return deserializeMaterial(getRegistry(), material_type, path, material_node);
}

}    // namespace MaterialRegistry
}    // namespace atcg

#define ATCG_REGISTER_MATERIAL(registry, MaterialType, MaterialClass)                                                  \
    {                                                                                                                  \
        atcg::MaterialRegistry::MaterialFunctions functions = {                                                        \
            [](const atcg::Dictionary& dict) { return atcg::make_ref<MaterialClass>(); },                              \
            [](const atcg::ref_ptr<atcg::Material>& material, const std::string& key, bool& deactivated)               \
            {                                                                                                          \
                return atcg::MaterialGUIRenderer<MaterialClass>::renderGUI(                                            \
                    std::dynamic_pointer_cast<MaterialClass>(material),                                                \
                    key,                                                                                               \
                    deactivated);                                                                                      \
            },                                                                                                         \
            [](const atcg::ref_ptr<atcg::Material>& material, const std::filesystem::path& path)                       \
            {                                                                                                          \
                atcg::MaterialSerializer<MaterialClass>::serialize(std::dynamic_pointer_cast<MaterialClass>(material), \
                                                                   path);                                              \
            },                                                                                                         \
            [](const std::filesystem::path& path, const nlohmann::json& material_node)                                 \
            { return atcg::MaterialSerializer<MaterialClass>::deserialize(path, material_node); }};                    \
        registry->registerType(MaterialType, std::move(functions));                                                    \
    }

#define ATCG_REGISTER_MATERIAL_PLUGIN(registry, handle, MaterialType, MaterialClass)                                   \
    {                                                                                                                  \
        atcg::MaterialRegistry::MaterialFunctions functions = {                                                        \
            [](const atcg::Dictionary& dict) { return atcg::make_ref<MaterialClass>(); },                              \
            [](const atcg::ref_ptr<atcg::Material>& material, const std::string& key, bool& deactivated)               \
            {                                                                                                          \
                return atcg::MaterialGUIRenderer<MaterialClass>::renderGUI(                                            \
                    std::dynamic_pointer_cast<MaterialClass>(material),                                                \
                    key,                                                                                               \
                    deactivated);                                                                                      \
            },                                                                                                         \
            [](const atcg::ref_ptr<atcg::Material>& material, const std::filesystem::path& path)                       \
            {                                                                                                          \
                atcg::MaterialSerializer<MaterialClass>::serialize(std::dynamic_pointer_cast<MaterialClass>(material), \
                                                                   path);                                              \
            },                                                                                                         \
            [](const std::filesystem::path& path, const nlohmann::json& material_node)                                 \
            { return atcg::MaterialSerializer<MaterialClass>::deserialize(path, material_node); }};                    \
        registry->registerType(handle, MaterialType, std::move(functions));                                            \
    }