#pragma once

#include <BSDF/BSDF.h>
#include <BSDF/DielectricBSDFData.cuh>
#include <Renderer/Texture.h>

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

    virtual std::vector<torch::Tensor> getParameterGradients() const override;

    virtual void zeroGrad() override;

    virtual void markOptimizable() override;

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

private:
    torch::Tensor _diffuse_texture;
    torch::Tensor _roughness_texture;
    torch::Tensor _ior_texture;

    atcg::dref_ptr<DielectricBSDFData> _bsdf_data_buffer;

    atcg::ref_ptr<Texture2D> _diffuse_optimized, _roughness_optimized, _ior_optimized;
    atcg::ref_ptr<Texture2D> _diffuse_grad, _roughness_grad, _ior_grad;
};
}    // namespace atcg