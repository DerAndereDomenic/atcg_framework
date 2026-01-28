#pragma once

#include <BSDF/BSDF.h>
#include <BSDF/DielectricBSDFData.cuh>
#include <Renderer/Texture.h>

namespace atcg
{
class DielectricBSDF : public BSDF
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

private:
    atcg::ref_ptr<Texture2D> _diffuse_texture;
    atcg::ref_ptr<Texture2D> _roughness_texture;
    atcg::ref_ptr<Texture2D> _ior_texture;

    atcg::dref_ptr<DielectricBSDFData> _bsdf_data_buffer;
};
}    // namespace atcg