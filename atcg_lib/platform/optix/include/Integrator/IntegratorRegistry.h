#pragma once

#include <Core/Assert.h>
#include <Core/SystemRegistry.h>
#include <DataStructure/Registry.h>
#include <Integrator/Integrator.h>
#include <Integrator/IntegratorSettings.h>

#include <functional>

namespace atcg
{

namespace IntegratorRegistry
{

using IntegratorBuilder =
    std::function<atcg::ref_ptr<Integrator>(const atcg::ref_ptr<RaytracingContext>&, const atcg::Dictionary&)>;
using SettingsBuilder = std::function<atcg::ref_ptr<IntegratorSettingsBase>()>;

struct IntegratorFunctions
{
    IntegratorBuilder builder;
    SettingsBuilder settings_builder;
};
using Registry = atcg::Registry<IntegratorFunctions>;

ATCG_API void registerIntegrator(Registry* registry, std::string_view type, IntegratorFunctions functions);

ATCG_API atcg::ref_ptr<Integrator> createIntegrator(Registry* registry,
                                                    const std::string& type,
                                                    const atcg::ref_ptr<RaytracingContext>& context,
                                                    const atcg::Dictionary& dict);

ATCG_API atcg::ref_ptr<IntegratorSettingsBase> createIntegratorSettings(Registry* registry, const std::string& type);

ATCG_INLINE Registry* getRegistry()
{
    Registry* registry = SystemRegistry::instance()->getSystem<IntegratorRegistry::Registry>();
    ATCG_ASSERT(registry, "Integrator registry not found");
    return registry;
}

ATCG_INLINE void registerIntegrator(std::string_view type, IntegratorFunctions functions)
{
    registerIntegrator(getRegistry(), type, std::move(functions));
}

ATCG_INLINE atcg::ref_ptr<Integrator>
createIntegrator(const std::string& type, const atcg::ref_ptr<RaytracingContext>& context, const atcg::Dictionary& dict)
{
    return createIntegrator(getRegistry(), type, context, dict);
}

ATCG_INLINE atcg::ref_ptr<IntegratorSettingsBase> createIntegratorSettings(const std::string& type)
{
    return createIntegratorSettings(getRegistry(), type);
}
}    // namespace IntegratorRegistry

}    // namespace atcg

#define ATCG_REGISTER_INTEGRATOR(registry, IntegratorType, IntegratorClass)                                            \
    {                                                                                                                  \
        atcg::IntegratorRegistry::IntegratorFunctions functions = {                                                    \
            [](const atcg::ref_ptr<atcg::RaytracingContext>& context,                                                  \
               const atcg::Dictionary& dict) -> atcg::ref_ptr<atcg::Integrator>                                        \
            { return atcg::make_ref<IntegratorClass>(context, dict); },                                                \
            []() -> atcg::ref_ptr<atcg::IntegratorSettingsBase>                                                        \
            { return atcg::make_ref<atcg::IntegratorSettings<IntegratorClass>>(); }};                                  \
        registry->registerType(IntegratorType, std::move(functions));                                                  \
    }

#define ATCG_REGISTER_INTEGRATOR_PLUGIN(registry, handle, IntegratorType, IntegratorClass)                             \
    {                                                                                                                  \
        atcg::IntegratorRegistry::IntegratorFunctions functions = {                                                    \
            [](const atcg::ref_ptr<atcg::RaytracingContext>& context,                                                  \
               const atcg::Dictionary& dict) -> atcg::ref_ptr<atcg::Integrator>                                        \
            { return atcg::make_ref<IntegratorClass>(context, dict); },                                                \
            []() -> atcg::ref_ptr<atcg::IntegratorSettingsBase>                                                        \
            { return atcg::make_ref<atcg::IntegratorSettings<IntegratorClass>>(); }};                                  \
        registry->registerType(handle, IntegratorType, std::move(functions));                                          \
    }