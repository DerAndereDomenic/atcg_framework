#pragma once

#include <Medium/PhaseFunction.h>
#include <Medium/HenyeyGreensteinPhaseFunctionData.cuh>

#include <Core/PipelineInitializer.h>

namespace atcg
{
/**
 * @brief Henyey Greenstein Phase function
 */
class HenyeyGreensteinPhaseFunction : public PhaseFunction
{
public:
    /**
     * @brief Constructor
     * Input parameters:
     * - "g": float
     *
     * @param dict Dictionary holding the parameters
     */
    HenyeyGreensteinPhaseFunction(const Dictionary& dict);

    /**
     * @brief Destructor
     */
    virtual ~HenyeyGreensteinPhaseFunction();

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() override {}

    ATCG_INLINE atcg::dref_ptr<HenyeyGreensteinPhaseFunctionData> getDataBuffer() const { return _data_buffer; }

private:
    atcg::dref_ptr<HenyeyGreensteinPhaseFunctionData> _data_buffer;
};

ATCG_DECLARE_COMPONENT_PIPELINE_INITIALIZER(HenyeyGreensteinPhaseFunction);
}    // namespace atcg