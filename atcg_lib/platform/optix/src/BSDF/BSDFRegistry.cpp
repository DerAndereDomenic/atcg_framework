#include <BSDF/BSDFRegistry.h>

namespace atcg
{

namespace BSDFRegistry
{
void registerBSDF(Registry* registry, std::string_view type, BSDFFunctions functions)
{
    registry->registerType(type, std::move(functions));
}

atcg::ref_ptr<BSDF> createBSDF(Registry* registry,
                               const std::string& type,
                               const Dictionary& dict,
                               const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                               const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const BSDFFunctions* functions = registry->find(type);
    if(!functions)
    {
        throw std::runtime_error("No BSDF registered for material type '" + type + "'.");
    }

    auto bsdf = functions->builder(dict);
    bsdf->initializePipeline(pipeline, sbt);

    return bsdf;
}

void validateMaterialCoverage()
{
    auto system_registry = SystemRegistry::instance();
    if(!system_registry->hasSystem<Registry>() || !system_registry->hasSystem<MaterialRegistry::Registry>())
    {
        return;
    }

    const Registry* bsdf_registry                       = system_registry->getSystem<Registry>();
    const MaterialRegistry::Registry* material_registry = system_registry->getSystem<MaterialRegistry::Registry>();

    for(const auto& material_type: material_registry->getRegisteredTypes())
    {
        if(bsdf_registry->find(material_type) == nullptr)
        {
            ATCG_WARN("Material type '{0}' has no BSDF registered under the same name. Objects with this material "
                      "cannot be path traced.",
                      material_type);
        }
    }
}
}    // namespace BSDFRegistry

}    // namespace atcg