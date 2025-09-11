#pragma once

#include <Core/Memory.h>
#include <BSDF/BSDF.h>
#include <DataStructure/Dictionary.h>
#include <Renderer/Material.h>

#include <unordered_map>
#include <functional>

namespace atcg
{
using BSDFBuilder = std::function<atcg::ref_ptr<BSDF>(const Dictionary&)>;

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
 *
 * @return The BSDF
 */
atcg::ref_ptr<BSDF> createBSDF(MaterialType type, const Dictionary& dict);
}    // namespace BSDFFactory

#define ATCG_REGISTER_BSDF(MaterialType, BSDFClass)                                                                    \
    struct BSDFFactory_##BSDFClass                                                                                     \
    {                                                                                                                  \
        BSDFFactory_##BSDFClass()                                                                                      \
        {                                                                                                              \
            BSDFFactory::registerBSDF(MaterialType,                                                                    \
                                      [](const Dictionary& dict) { return atcg::make_ref<BSDFClass>(dict); });         \
        }                                                                                                              \
        static BSDFFactory_##BSDFClass instance;                                                                       \
    };                                                                                                                 \
    BSDFFactory_##BSDFClass BSDFFactory_##BSDFClass::instance

}    // namespace atcg