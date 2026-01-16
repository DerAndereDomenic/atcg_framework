#pragma once

#include <BSDF/BSDF.h>
#include <BSDF/DielectricBSDFData.cuh>
#include <Renderer/Texture.h>
#include <Core/PipelineInitializer.h>

namespace atcg
{
class DielectricBSDF : public BSDF, public Differentiable
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

    virtual std::vector<torch::Tensor> getParameters() const override;

    virtual void markOptimizable() override;

    virtual void clampParameters() override;

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() override;

    ATCG_INLINE atcg::dref_ptr<DielectricBSDFData> getDataBuffer() const { return _bsdf_data_buffer; }

private:
    torch::Tensor _diffuse_texture;
    torch::Tensor _roughness_texture;
    torch::Tensor _ior_texture;

    atcg::dref_ptr<DielectricBSDFData> _bsdf_data_buffer;

    atcg::ref_ptr<Texture2D> _diffuse_optimized, _roughness_optimized, _ior_optimized;
    atcg::ref_ptr<Texture2D> _diffuse_grad, _roughness_grad, _ior_grad;
};

ATCG_DECLARE_COMPONENT_PIPELINE_INITIALIZER(DielectricBSDF);
}    // namespace atcg