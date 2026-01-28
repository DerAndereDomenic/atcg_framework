#pragma once

#include <Core/Memory.h>
#include <BSDF/BSDF.h>
#include <DataStructure/Dictionary.h>
#include <Renderer/Material.h>
#include <Core/RaytracingPipeline.h>
#include <Core/ShaderBindingTable.h>

#include <unordered_map>
#include <functional>

namespace atcg
{
using BSDFBuilder = std::function<atcg::ref_ptr<BSDF>(const Dictionary&,
                                                      const atcg::ref_ptr<RayTracingPipeline>&,
                                                      const atcg::ref_ptr<ShaderBindingTable>&)>;

namespace BSDFFactory
{
/**
 * @brief Register a BSDF type
 *
 * @param type The material type
 * @param builder The builder
 */
void registerBSDF(MaterialType type, BSDFBuilder builder);

/**
 * @brief Create a BSDF based on the material type
 *
 * @param type The material type
 * @param dict Parameters
 * @param pipeline The raytracing pipeline
 * @param sbt The shader binding table
 *
 * @return The BSDF
 */
atcg::ref_ptr<BSDF> createBSDF(MaterialType type,
                               const Dictionary& dict,
                               const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                               const atcg::ref_ptr<ShaderBindingTable>& sbt);
}    // namespace BSDFFactory

#define ATCG_REGISTER_BSDF(MaterialType, BSDFClass)                                                                    \
    struct BSDFFactory_##BSDFClass                                                                                     \
    {                                                                                                                  \
        BSDFFactory_##BSDFClass()                                                                                      \
        {                                                                                                              \
            BSDFFactory::registerBSDF(MaterialType,                                                                    \
                                      [](const Dictionary& dict,                                                       \
                                         const atcg::ref_ptr<RayTracingPipeline>& pipeline,                            \
                                         const atcg::ref_ptr<ShaderBindingTable>& sbt)                                 \
                                      {                                                                                \
                                          auto bsdf = atcg::make_ref<BSDFClass>(dict);                                 \
                                          bsdf->initializePipeline(pipeline, sbt);                                     \
                                          return bsdf;                                                                 \
                                      });                                                                              \
        }                                                                                                              \
        static BSDFFactory_##BSDFClass instance;                                                                       \
    };                                                                                                                 \
    BSDFFactory_##BSDFClass BSDFFactory_##BSDFClass::instance

}    // namespace atcg