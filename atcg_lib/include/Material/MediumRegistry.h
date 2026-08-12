#pragma once

#include <Core/Assert.h>
#include <Material/Medium.h>
#include <DataStructure/Registry.h>

namespace atcg
{
namespace MediumRegistry
{
using MediumBuilder           = std::function<atcg::ref_ptr<Medium>(const Dictionary&)>;
using MediumGUIFunction       = std::function<bool(const atcg::ref_ptr<Medium>&, const std::string&, bool&)>;
using MediumSerializeFunction = std::function<void(const atcg::ref_ptr<Medium>&, const std::filesystem::path&)>;
using MediumDeserializeFunction =
    std::function<atcg::ref_ptr<Medium>(const std::filesystem::path&, const nlohmann::json&)>;

struct MediumFunctions
{
    MediumBuilder builder;
    MediumGUIFunction gui_function;
    MediumSerializeFunction serializer_function;
    MediumDeserializeFunction deserializer_function;
};

using Registry = atcg::Registry<MediumFunctions>;

ATCG_API void registerMedium(Registry* registry, std::string_view type, MediumFunctions);

ATCG_API atcg::ref_ptr<Medium> createMedium(Registry* registry, const std::string& type, const Dictionary& dict);

ATCG_API bool
renderMediumGUI(Registry* registry, const atcg::ref_ptr<Medium>& medium, const std::string& key, bool& deactivated);

ATCG_API const std::vector<std::string>& getRegisteredMediumTypes(Registry* registry);

ATCG_API void
serializeMedium(Registry* registry, const atcg::ref_ptr<Medium>& medium, const std::filesystem::path& path);

ATCG_API atcg::ref_ptr<Medium> deserializeMedium(Registry* registry,
                                                 std::string_view medium_type,
                                                 const std::filesystem::path& path,
                                                 const nlohmann::json& medium_node);

ATCG_INLINE Registry* getRegistry()
{
    Registry* registry = SystemRegistry::instance()->getSystem<MediumRegistry::Registry>();
    ATCG_ASSERT(registry, "Medium registry not found");
    return registry;
}

ATCG_INLINE void registerMedium(std::string_view type, MediumFunctions functions)
{
    registerMedium(getRegistry(), type, std::move(functions));
}

ATCG_INLINE atcg::ref_ptr<Medium> createMedium(const std::string& type, const Dictionary& dict)
{
    return createMedium(getRegistry(), type, dict);
}

ATCG_INLINE bool renderMediumGUI(const atcg::ref_ptr<Medium>& medium, const std::string& key, bool& deactivated)
{
    return renderMediumGUI(getRegistry(), medium, key, deactivated);
}

ATCG_INLINE const std::vector<std::string>& getRegisteredMediumTypes()
{
    return getRegisteredMediumTypes(getRegistry());
}

ATCG_INLINE void serializeMedium(const atcg::ref_ptr<Medium>& medium, const std::filesystem::path& path)
{
    serializeMedium(getRegistry(), medium, path);
}

ATCG_INLINE atcg::ref_ptr<Medium>
deserializeMedium(std::string_view medium_type, const std::filesystem::path& path, const nlohmann::json& medium_node)
{
    return deserializeMedium(getRegistry(), medium_type, path, medium_node);
}

}    // namespace MediumRegistry
}    // namespace atcg

#define ATCG_REGISTER_MEDIUM(registry, MediumType, MediumClass)                                                        \
    {                                                                                                                  \
        atcg::MediumRegistry::MediumFunctions functions = {                                                            \
            [](const atcg::Dictionary& dict) { return atcg::make_ref<MediumClass>(dict); },                            \
            [](const atcg::ref_ptr<atcg::Medium>& medium, const std::string& key, bool& deactivated)                   \
            {                                                                                                          \
                return atcg::MediumGUIRenderer<MediumClass>::renderGUI(std::dynamic_pointer_cast<MediumClass>(medium), \
                                                                       key,                                            \
                                                                       deactivated);                                   \
            },                                                                                                         \
            [](const atcg::ref_ptr<atcg::Medium>& medium, const std::filesystem::path& path)                           \
            { atcg::MediumSerializer<MediumClass>::serialize(std::dynamic_pointer_cast<MediumClass>(medium), path); }, \
            [](const std::filesystem::path& path, const nlohmann::json& material_node)                                 \
            { return atcg::MediumSerializer<MediumClass>::deserialize(path, material_node); }};                        \
        registry->registerType(MediumType, std::move(functions));                                                      \
    }

#define ATCG_REGISTER_MEDIUM_PLUGIN(registry, handle, MediumType, MediumClass)                                         \
    {                                                                                                                  \
        atcg::MediumRegistry::MediumFunctions functions = {                                                            \
            [](const atcg::Dictionary& dict) { return atcg::make_ref<MediumClass>(dict); },                            \
            [](const atcg::ref_ptr<atcg::Medium>& medium, const std::string& key, bool& deactivated)                   \
            {                                                                                                          \
                return atcg::MediumGUIRenderer<MediumClass>::renderGUI(std::dynamic_pointer_cast<MediumClass>(medium), \
                                                                       key,                                            \
                                                                       deactivated);                                   \
            },                                                                                                         \
            [](const atcg::ref_ptr<atcg::Medium>& medium, const std::filesystem::path& path)                           \
            { atcg::MediumSerializer<MediumClass>::serialize(std::dynamic_pointer_cast<MediumClass>(medium), path); }, \
            [](const std::filesystem::path& path, const nlohmann::json& material_node)                                 \
            { return atcg::MediumSerializer<MediumClass>::deserialize(path, material_node); }};                        \
        registry->registerType(handle, MediumType, std::move(functions));                                              \
    }