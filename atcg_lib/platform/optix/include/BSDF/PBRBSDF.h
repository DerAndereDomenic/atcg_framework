#pragma once

#include <BSDF/BSDF.h>
#include <BSDF/PBRBSDFData.cuh>
#include <Renderer/Material.h>

namespace atcg
{
/**
 * @brief A PBR BSDF
 */
class PBRBSDF : public BSDF, public Differentiable
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

    virtual std::vector<torch::Tensor> getParameters() const override;

    virtual void markOptimizable() override;
    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() override;

    virtual void clampParameters() override;

    /**
     * @brief Initialize the pipeline
     *
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    torch::Tensor _diffuse_texture;
    torch::Tensor _metallic_texture;
    torch::Tensor _roughness_texture;

    atcg::dref_ptr<PBRBSDFData> _bsdf_data_buffer;

    atcg::ref_ptr<Texture2D> _diffuse_optimized, _metallic_optimized, _roughness_optimized;
    atcg::ref_ptr<Texture2D> _diffuse_grad, _metallic_grad, _roughness_grad;
};
}    // namespace atcg