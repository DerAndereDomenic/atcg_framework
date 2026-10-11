#include <Integrator/IntegratorRegistry.h>

namespace atcg
{
namespace IntegratorRegistry
{
void registerIntegrator(Registry* registry, std::string_view type, IntegratorFunctions functions)
{
    registry->registerType(type, std::move(functions));
}

atcg::ref_ptr<Integrator> createIntegrator(Registry* registry,
                                           const std::string& type,
                                           const atcg::ref_ptr<RaytracingContext>& context,
                                           const atcg::Dictionary& dict)
{
    const IntegratorFunctions* functions = registry->find(type);
    if(!functions)
    {
        ATCG_ERROR("Integrator type {} not found in registry", type);
        return nullptr;
    }
    return functions->builder(context, dict);
}

atcg::ref_ptr<IntegratorSettingsBase> createIntegratorSettings(Registry* registry, const std::string& type)
{
    const IntegratorFunctions* functions = registry->find(type);
    if(!functions)
    {
        ATCG_ERROR("Integrator type {} not found in registry", type);
        return nullptr;
    }
    return functions->settings_builder();
}
}    // namespace IntegratorRegistry
}    // namespace atcg