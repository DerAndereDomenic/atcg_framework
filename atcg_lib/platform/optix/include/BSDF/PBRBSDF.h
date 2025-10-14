#pragma once

#include <BSDF/BSDF.h>
#include <BSDF/PBRBSDFData.cuh>
#include <Renderer/Material.h>
#include <Core/PipelineInitializer.h>

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

    ATCG_INLINE atcg::dref_ptr<PBRBSDFData> getDataBuffer() const { return _bsdf_data_buffer; }

private:
    torch::Tensor _diffuse_texture;
    torch::Tensor _metallic_texture;
    torch::Tensor _roughness_texture;

    atcg::dref_ptr<PBRBSDFData> _bsdf_data_buffer;

    atcg::ref_ptr<Texture2D> _diffuse_optimized, _metallic_optimized, _roughness_optimized;
    atcg::ref_ptr<Texture2D> _diffuse_grad, _metallic_grad, _roughness_grad;
};

ATCG_DECLARE_COMPONENT_PIPELINE_INITIALIZER(PBRBSDF);
}    // namespace atcg