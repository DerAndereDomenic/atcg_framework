#pragma once

#include <BSDF/BSDF.h>
#include <BSDF/PBRBSDFData.cuh>
#include <Renderer/Material.h>
#include <DataStructure/Statistics.h>
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

    virtual void markParameterAsOptimizable(const const std::string& parameter_name) override;

    virtual void clampParameters() override;
    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() override;

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
    atcg::dref_ptr<PBRBSDFData> _bsdf_data_buffer;

    atcg::ref_ptr<Texture2D> _diffuse_optimized, _metallic_optimized, _roughness_optimized;
    atcg::ref_ptr<Texture2D> _diffuse_grad, _metallic_grad, _roughness_grad;

    uint32_t _optimization_width  = 1;
    uint32_t _optimization_height = 1;

    atcg::CyclicCollection<float> time_collection = atcg::CyclicCollection<float>("Time Collection", 35 * 60 / 5);
    atcg::CyclicCollection<float> roughness_collection =
        atcg::CyclicCollection<float>("Roughness Collection", 35 * 60 / 5);
    atcg::CyclicCollection<float> roughness_grad_collection =
        atcg::CyclicCollection<float>("Roughness grad Collection", 35 * 60 / 5);
};
}    // namespace atcg