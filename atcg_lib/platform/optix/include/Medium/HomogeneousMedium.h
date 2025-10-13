#pragma once

#include <Medium/Medium.h>
#include <Medium/HomogeneousMediumData.cuh>

#include <Scene/ComponentGUIHandler.h>
#include <Scene/ComponentSerializer.h>
#include <Core/PipelineInitializer.h>

namespace atcg
{
class HomogeneousMedium : public Medium
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

    ATCG_INLINE atcg::dref_ptr<HomogeneousMediumData> getDataBuffer() const { return _data_buffer; }

private:
    atcg::dref_ptr<HomogeneousMediumData> _data_buffer;
};

ATCG_DECLARE_COMPONENT_PIPELINE_INITIALIZER(HomogeneousMedium);

}    // namespace atcg