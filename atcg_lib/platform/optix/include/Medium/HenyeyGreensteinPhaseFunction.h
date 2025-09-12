#pragma once

#include <Medium/PhaseFunction.h>
#include <Medium/HenyeyGreensteinPhaseFunctionData.cuh>

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
     * @brief Initialize a pipeline.
     * This function should be overwritten by each child class and it should add its functions to the pipeline and the
     * sbt.
     *
     * @param pipeline The pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    atcg::dref_ptr<HenyeyGreensteinPhaseFunctionData> _data_buffer;
};
}    // namespace atcg