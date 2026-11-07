#pragma once

#include <BSDF/BSDF.h>
#include "PerfectDielectricData.cuh"
#include <Renderer/Texture.h>
#include <BSDF/BSDFRegistry.h>

namespace atcg
{
class PerfectDielectricBSDF : public BSDF
{
public:
    /**
     * @brief Construct a BSDF with arbitrary parameters.
     * Input parameters:
     * - "material": atcg::Material
     *
     * @param dict Dictionary holding the parameters
     */
    PerfectDielectricBSDF(const atcg::Dictionary& dict);

    /**
     * @brief Destructor
     */
    virtual ~PerfectDielectricBSDF();

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
    atcg::dref_ptr<PerfectDielectricData> _bsdf_data_buffer;

    torch::Tensor _ior_texture;
};
}    // namespace atcg