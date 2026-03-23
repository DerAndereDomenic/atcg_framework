#pragma once

#include <Medium/PhaseFunction.h>
#include <Medium/HenyeyGreensteinPhaseFunctionData.cuh>

#include <DataStructure/Statistics.h>

namespace atcg
{
/**
 * @brief Henyey Greenstein Phase function
 */
class HenyeyGreensteinPhaseFunction : public PhaseFunction, public Differentiable
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
    virtual void onImGuiRender() override;

    /**
     * @brief Initialize the pipeline
     *
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

    virtual std::vector<torch::Tensor> getParameters() const override;

    virtual std::vector<torch::Tensor> getParameterGradients() const override;

    virtual void zeroGrad() override;

    virtual void markOptimizable() override;

    virtual void clampParameters() override;

    ATCG_INLINE atcg::dref_ptr<HenyeyGreensteinPhaseFunctionData> getDataBuffer() const { return _data_buffer; }

private:
    atcg::dref_ptr<HenyeyGreensteinPhaseFunctionData> _data_buffer;

    torch::Tensor _g_tensor;
    torch::Tensor _g_grad_tensor;

    bool _optimize_g = false;

    atcg::CyclicCollection<float> time_collection   = atcg::CyclicCollection<float>("Time Collection", 35 * 60 / 5);
    atcg::CyclicCollection<float> g_collection      = atcg::CyclicCollection<float>("g Collection", 35 * 60 / 5);
    atcg::CyclicCollection<float> g_grad_collection = atcg::CyclicCollection<float>("g grad Collection", 35 * 60 / 5);
};
}    // namespace atcg