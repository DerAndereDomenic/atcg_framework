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
    ~PBRBSDF();

    /**
     * @brief Initialize a pipeline.
     * This function should be overwritten by each child class and it should add its functions to the pipeline and the
     * sbt.
     *
     * @param pipeline The pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

    virtual void zero_grad() override;

private:
    atcg::ref_ptr<Texture2D> _diffuse_texture;
    atcg::ref_ptr<Texture2D> _metallic_texture;
    atcg::ref_ptr<Texture2D> _roughness_texture;

    torch::Tensor _grad_diffuse;
    torch::Tensor _grad_metallic;
    torch::Tensor _grad_roughness;

    atcg::dref_ptr<PBRBSDFData> _bsdf_data_buffer;
};
}    // namespace atcg