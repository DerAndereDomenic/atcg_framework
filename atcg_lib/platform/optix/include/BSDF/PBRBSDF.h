#pragma once

#include <BSDF/BSDF.h>
#include <BSDF/PBRBSDFData.cuh>
#include <Material/Material.h>
#include <BSDF/BSDFRegistry.h>

namespace atcg
{
/**
 * @brief A PBR BSDF
 */
class ATCG_API PBRBSDF : public BSDF
{
public:
    /**
     * @brief Construct a BSDF with arbitrary parameters.
     * Input parameters:
     * - "material": atcg::Material
     *
     * @param dict Dictionary holding the parameters
     */
    PBRBSDF(const atcg::Dictionary& dict);

    /**
     * @brief Destructor
     */
    virtual ~PBRBSDF();

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() override {};

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
    atcg::ref_ptr<Texture2D> _diffuse_texture;
    atcg::ref_ptr<Texture2D> _metallic_texture;
    atcg::ref_ptr<Texture2D> _roughness_texture;

    atcg::dref_ptr<PBRBSDFData> _bsdf_data_buffer;
};
}    // namespace atcg