#pragma once

#include <Medium/Medium.h>
#include <Medium/HomogeneousMediumData.cuh>

#include <Scene/ComponentGUIHandler.h>
#include <Scene/ComponentSerializer.h>
#include <DataStructure/Statistics.h>

namespace atcg
{
class HomogeneousMedium : public Medium, public Differentiable
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

private:
    atcg::dref_ptr<HomogeneousMediumData> _data_buffer;

    torch::Tensor _albedo_tensor;
    torch::Tensor _density_tensor;

    torch::Tensor _albedo_grad_tensor;
    torch::Tensor _density_grad_tensor;

    bool _optimize_albedo  = false;
    bool _optimize_density = false;

    atcg::CyclicCollection<float> time_collection    = atcg::CyclicCollection<float>("Time Collection", 35 * 60 / 5);
    atcg::CyclicCollection<float> density_collection = atcg::CyclicCollection<float>("Density Collection", 35 * 60 / 5);
    atcg::CyclicCollection<float> density_grad_collection =
        atcg::CyclicCollection<float>("Density grad Collection", 35 * 60 / 5);
};

}    // namespace atcg