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
        throw std::runtime_error("Unknown BSDF type");
    }

    auto bsdf = functions->builder(dict);
    bsdf->initializePipeline(pipeline, sbt);

    return bsdf;
}
}    // namespace BSDFRegistry

}    // namespace atcg