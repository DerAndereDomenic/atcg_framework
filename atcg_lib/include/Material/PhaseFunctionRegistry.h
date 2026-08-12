#pragma once

#include <Material/PhaseFunction.h>
#include <DataStructure/Registry.h>

namespace atcg
{

namespace PhaseFunctionRegistry
{
using PhaseFunctionBuilder     = std::function<atcg::ref_ptr<PhaseFunction>(const Dictionary&)>;
using PhaseFunctionGUIFunction = std::function<bool(const atcg::ref_ptr<PhaseFunction>&, const std::string&, bool&)>;
using PhaseFunctionSerializeFunction =
    std::function<void(const atcg::ref_ptr<PhaseFunction>&, const std::filesystem::path&)>;
using PhaseFunctionDeserializeFunction =
    std::function<atcg::ref_ptr<PhaseFunction>(const std::filesystem::path&, const nlohmann::json&)>;
struct PhaseFunctionFunctions
{
    PhaseFunctionBuilder builder;
    PhaseFunctionGUIFunction gui_function;
    PhaseFunctionSerializeFunction serializer_function;
    PhaseFunctionDeserializeFunction deserializer_function;
};

using Registry = atcg::Registry<PhaseFunctionFunctions>;

ATCG_API void registerPhaseFunction(Registry* registry, std::string_view type, PhaseFunctionFunctions);

ATCG_API atcg::ref_ptr<PhaseFunction>
createPhaseFunction(Registry* registry, const std::string& type, const Dictionary& dict);

ATCG_API bool renderPhaseFunctionGUI(Registry* registry,
                                     const atcg::ref_ptr<PhaseFunction>& phase_function,
                                     const std::string& key,
                                     bool& deactivated);

ATCG_API const std::vector<std::string>& getRegisteredPhaseFunctionTypes(Registry* registry);

ATCG_API void serializePhaseFunction(Registry* registry,
                                     const atcg::ref_ptr<PhaseFunction>& phase_function,
                                     const std::filesystem::path& path);

ATCG_API atcg::ref_ptr<PhaseFunction> deserializePhaseFunction(Registry* registry,
                                                               std::string_view phase_function_type,
                                                               const std::filesystem::path& path,
                                                               const nlohmann::json& phase_function_node);

ATCG_INLINE Registry* getRegistry()
{
    Registry* registry = SystemRegistry::instance()->getSystem<PhaseFunctionRegistry::Registry>();
    ATCG_ASSERT(registry, "Phase function registry not found");
    return registry;
}

ATCG_INLINE void registerPhaseFunction(std::string_view type, PhaseFunctionFunctions functions)
{
    registerPhaseFunction(getRegistry(), type, std::move(functions));
}

ATCG_INLINE atcg::ref_ptr<PhaseFunction> createPhaseFunction(const std::string& type, const Dictionary& dict)
{
    return createPhaseFunction(getRegistry(), type, dict);
}

ATCG_INLINE bool
renderPhaseFunctionGUI(const atcg::ref_ptr<PhaseFunction>& phase_function, const std::string& key, bool& deactivated)
{
    return renderPhaseFunctionGUI(getRegistry(), phase_function, key, deactivated);
}

ATCG_INLINE const std::vector<std::string>& getRegisteredPhaseFunctionTypes()
{
    return getRegisteredPhaseFunctionTypes(getRegistry());
}

ATCG_INLINE void serializePhaseFunction(const atcg::ref_ptr<PhaseFunction>& phase_function,
                                        const std::filesystem::path& path)
{
    serializePhaseFunction(getRegistry(), phase_function, path);
}

ATCG_INLINE atcg::ref_ptr<PhaseFunction> deserializePhaseFunction(std::string_view phase_function_type,
                                                                  const std::filesystem::path& path,
                                                                  const nlohmann::json& phase_function_node)
{
    return deserializePhaseFunction(getRegistry(), phase_function_type, path, phase_function_node);
}

}    // namespace PhaseFunctionRegistry
}    // namespace atcg

#define ATCG_REGISTER_PHASE_FUNCTION(registry, PhaseFunctionType, PhaseFunctionClass)                                  \
    {                                                                                                                  \
        atcg::PhaseFunctionRegistry::PhaseFunctionFunctions functions = {                                              \
            [](const atcg::Dictionary& dict) { return atcg::make_ref<PhaseFunctionClass>(dict); },                     \
            [](const atcg::ref_ptr<atcg::PhaseFunction>& phase_function, const std::string& key, bool& deactivated)    \
            {                                                                                                          \
                return atcg::PhaseFunctionGUIRenderer<PhaseFunctionClass>::renderGUI(                                  \
                    std::dynamic_pointer_cast<PhaseFunctionClass>(phase_function),                                     \
                    key,                                                                                               \
                    deactivated);                                                                                      \
            },                                                                                                         \
            [](const atcg::ref_ptr<atcg::PhaseFunction>& phase_function, const std::filesystem::path& path)            \
            {                                                                                                          \
                atcg::PhaseFunctionSerializer<PhaseFunctionClass>::serialize(                                          \
                    std::dynamic_pointer_cast<PhaseFunctionClass>(phase_function),                                     \
                    path);                                                                                             \
            },                                                                                                         \
            [](const std::filesystem::path& path, const nlohmann::json& phase_function_node)                           \
            { return atcg::PhaseFunctionSerializer<PhaseFunctionClass>::deserialize(path, phase_function_node); }};    \
        registry->registerType(PhaseFunctionType, std::move(functions));                                               \
    }

#define ATCG_REGISTER_PHASE_FUNCTION_PLUGIN(registry, handle, PhaseFunctionType, PhaseFunctionClass)                   \
    {                                                                                                                  \
        atcg::PhaseFunctionRegistry::PhaseFunctionFunctions functions = {                                              \
            [](const atcg::Dictionary& dict) { return atcg::make_ref<PhaseFunctionClass>(dict); },                     \
            [](const atcg::ref_ptr<atcg::PhaseFunction>& phase_function, const std::string& key, bool& deactivated)    \
            {                                                                                                          \
                return atcg::PhaseFunctionGUIRenderer<PhaseFunctionClass>::renderGUI(                                  \
                    std::dynamic_pointer_cast<PhaseFunctionClass>(phase_function),                                     \
                    key,                                                                                               \
                    deactivated);                                                                                      \
            },                                                                                                         \
            [](const atcg::ref_ptr<atcg::PhaseFunction>& phase_function, const std::filesystem::path& path)            \
            {                                                                                                          \
                atcg::PhaseFunctionSerializer<PhaseFunctionClass>::serialize(                                          \
                    std::dynamic_pointer_cast<PhaseFunctionClass>(phase_function),                                     \
                    path);                                                                                             \
            },                                                                                                         \
            [](const std::filesystem::path& path, const nlohmann::json& phase_function_node)                           \
            { return atcg::PhaseFunctionSerializer<PhaseFunctionClass>::deserialize(path, phase_function_node); }};    \
        registry->registerType(handle, PhaseFunctionType, std::move(functions));                                       \
    }