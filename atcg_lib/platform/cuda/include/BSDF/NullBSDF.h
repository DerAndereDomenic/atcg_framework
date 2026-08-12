#pragma once

#include <BSDF/BSDF.h>
#include <Material/Material.h>
#include <BSDF/BSDFRegistry.h>

namespace atcg
{
/**
 * @brief A PBR BSDF
 */
class ATCG_API NullBSDF : public BSDF
{
public:
    /**
     * @brief Construct a Null BSDF
     *
     * @param dict Dictionary holding the parameters (unused)
     */
    NullBSDF(const atcg::Dictionary& dict);

    /**
     * @brief Destructor
     */
    virtual ~NullBSDF();

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() override {}

    /**
     * @brief Initialize the pipeline
     *
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

    static void registerBSDF(BSDFRegistry::Registry* registry);

private:
};

}    // namespace atcg