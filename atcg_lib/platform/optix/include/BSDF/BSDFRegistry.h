#pragma once

#include <Core/Memory.h>
#include <BSDF/BSDF.h>
#include <DataStructure/Dictionary.h>
#include <Renderer/Material.h>
#include <Core/RaytracingPipeline.h>
#include <Core/ShaderBindingTable.h>
#include <DataStructure/Registry.h>

#include <unordered_map>
#include <functional>

namespace atcg
{
using BSDFBuilder = std::function<atcg::ref_ptr<BSDF>(const Dictionary&)>;

namespace BSDFRegistry
{
struct BSDFFunctions
{
    BSDFBuilder builder;
};

using Registry = atcg::Registry<BSDFFunctions>;
ATCG_API void registerBSDF(Registry* registry, std::string_view type, BSDFFunctions functions);

ATCG_API atcg::ref_ptr<BSDF> createBSDF(Registry* registry,
                                        const std::string& type,
                                        const Dictionary& dict,
                                        const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                        const atcg::ref_ptr<ShaderBindingTable>& sbt);

ATCG_INLINE Registry* getRegistry()
{
    Registry* registry = SystemRegistry::instance()->getSystem<BSDFRegistry::Registry>();
    ATCG_ASSERT(registry, "BSDF registry not found");
    return registry;
}

ATCG_INLINE void registerBSDF(std::string_view type, BSDFFunctions functions)
{
    registerBSDF(getRegistry(), type, std::move(functions));
}

ATCG_INLINE atcg::ref_ptr<BSDF> createBSDF(const std::string& type,
                                           const Dictionary& dict,
                                           const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                           const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    return createBSDF(getRegistry(), type, dict, pipeline, sbt);
}
}    // namespace BSDFRegistry

#define ATCG_REGISTER_BSDF(registry, MaterialType, BSDFClass)                                                          \
    {                                                                                                                  \
        atcg::BSDFRegistry::BSDFFunctions functions = {[](const atcg::Dictionary& dict) -> atcg::ref_ptr<BSDF>         \
                                                       { return atcg::make_ref<BSDFClass>(dict); }};                   \
        registry->registerType(MaterialType, std::move(functions));                                                    \
    }

#define ATCG_REGISTER_BSDF_PLUGIN(registry, handle, MaterialType, BSDFClass)                                           \
    {                                                                                                                  \
        atcg::BSDFRegistry::BSDFFunctions functions = {[](const atcg::Dictionary& dict) -> atcg::ref_ptr<BSDF>         \
                                                       { return atcg::make_ref<BSDFClass>(dict); }};                   \
        registry->registerType(handle, MaterialType, std::move(functions));                                            \
    }

}    // namespace atcg