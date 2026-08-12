#include <Material/PhaseFunctionRegistry.h>

namespace atcg
{
namespace PhaseFunctionRegistry
{
void registerPhaseFunction(Registry* registry, std::string_view type, PhaseFunctionFunctions functions)
{
    registry->registerType(type, std::move(functions));
}

atcg::ref_ptr<PhaseFunction> createPhaseFunction(Registry* registry, const std::string& type, const Dictionary& dict)
{
    const PhaseFunctionFunctions* desc = registry->find(type);
    return desc->builder(dict);
}

bool renderPhaseFunctionGUI(Registry* registry,
                            const atcg::ref_ptr<PhaseFunction>& phase_function,
                            const std::string& key,
                            bool& deactivated)
{
    const PhaseFunctionFunctions* desc = registry->find(phase_function->getPhaseFunctionType());
    return desc->gui_function(phase_function, key, deactivated);
}

const std::vector<std::string>& getRegisteredPhaseFunctionTypes(Registry* registry)
{
    return registry->getRegisteredTypes();
}

void serializePhaseFunction(Registry* registry,
                            const atcg::ref_ptr<PhaseFunction>& phase_function,
                            const std::filesystem::path& path)
{
    const PhaseFunctionFunctions* desc = registry->find(phase_function->getPhaseFunctionType());
    desc->serializer_function(phase_function, path);
}

atcg::ref_ptr<PhaseFunction> deserializePhaseFunction(Registry* registry,
                                                      std::string_view phase_function_type,
                                                      const std::filesystem::path& path,
                                                      const nlohmann::json& phase_function_node)
{
    const PhaseFunctionFunctions* desc = registry->find(phase_function_type);
    return desc->deserializer_function(path, phase_function_node);
}
}    // namespace PhaseFunctionRegistry
}    // namespace atcg