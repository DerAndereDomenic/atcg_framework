#pragma once

#include <BSDF/BSDF.h>
#include <BSDF/DielectricBSDFData.cuh>
#include <Renderer/Texture.h>
#include <BSDF/BSDFRegistry.h>

namespace atcg
{
class ATCG_API DielectricBSDF : public BSDF, public Differentiable
{
public:
    /**
     * @brief Construct a BSDF with arbitrary parameters.
     * Input parameters:
     * - "material": atcg::Material
     *
     * @param dict Dictionary holding the parameters
     */
    DielectricBSDF(const atcg::Dictionary& dict);

    /**
     * @brief Destructor
     */
    virtual ~DielectricBSDF();

    virtual void markParametersAsOptimizable(const const std::string& parameter_name) override;

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
    atcg::dref_ptr<DielectricBSDFData> _bsdf_data_buffer;

    atcg::ref_ptr<Texture2D> _diffuse_optimized, _roughness_optimized, _ior_optimized;
    atcg::ref_ptr<Texture2D> _diffuse_grad, _roughness_grad, _ior_grad;
};
}    // namespace atcg