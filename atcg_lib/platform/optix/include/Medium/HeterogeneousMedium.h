#pragma once

#include <Core/Memory.h>
#include <Medium/Medium.h>
#include <Medium/HeterogeneousMediumData.cuh>
#include <DataStructure/Dictionary.h>
#include <Renderer/Texture.h>
#include <Asset/AssetManagerSystem.h>

#include <Scene/ComponentGUIHandler.h>
#include <Scene/ComponentSerializer.h>

namespace atcg
{
class HeterogeneousMedium : public Medium, public Differentiable
{
public:
    HeterogeneousMedium(const Dictionary& dict);

    virtual ~HeterogeneousMedium();
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
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt);

    virtual std::vector<torch::Tensor> getParameters() const override;

    virtual std::vector<torch::Tensor> getParameterGradients() const override;

    virtual void zeroGrad() override;

    virtual void markOptimizable() override;

    virtual void clampParameters() override;

private:
    torch::Tensor _albedo_tensor;
    torch::Tensor _density_tensor;
    torch::Tensor _emission_tensor;

    torch::Tensor _albedo_grad_tensor;
    torch::Tensor _density_grad_tensor;

    atcg::ref_ptr<Texture2D> _density_texture;
    atcg::ref_ptr<Texture2D> _density_grad_texture;
    int _layer = 0;

    bool _optimize_albedo  = false;
    bool _optimize_density = false;

    atcg::dref_ptr<HeterogeneousMediumData> _data_buffer;
};

}    // namespace atcg