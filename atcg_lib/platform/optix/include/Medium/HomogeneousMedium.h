#pragma once

#include <Medium/Medium.h>
#include <Medium/HomogeneousMediumData.cuh>

#include <Scene/ComponentGUIHandler.h>
#include <Scene/ComponentSerializer.h>

namespace atcg
{
class ATCG_API HomogeneousMedium : public Medium
{
public:
    /**
     * @brief Constructor
     * Inputs:
     * - "sigma_s" : vec3
     * - "sigma_a" : vec3
     * - "phase_func" : atcg::ref_ptr<PhaseFunction>
     *
     * @param dict The input parameters
     */
    HomogeneousMedium(const atcg::Dictionary& dict);

    /**
     * @brief Destructor
     */
    virtual ~HomogeneousMedium();

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
    atcg::dref_ptr<HomogeneousMediumData> _data_buffer;
};

}    // namespace atcg