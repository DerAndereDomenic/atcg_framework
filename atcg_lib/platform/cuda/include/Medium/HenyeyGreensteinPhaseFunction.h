#pragma once

#include <Medium/PhaseFunction.h>
#include <Medium/HenyeyGreensteinPhaseFunctionData.cuh>

namespace atcg
{
/**
 * @brief Henyey Greenstein Phase function
 */
class ATCG_API HenyeyGreensteinPhaseFunction : public PhaseFunction
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

    /**
     * @brief Initialize the pipeline
     *
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

    ATCG_INLINE atcg::dref_ptr<HenyeyGreensteinPhaseFunctionData> getDataBuffer() const { return _data_buffer; }

private:
    atcg::dref_ptr<HenyeyGreensteinPhaseFunctionData> _data_buffer;
};
}    // namespace atcg