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
class ATCG_API HeterogeneousMedium : public Medium
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

    virtual void markParametersAsOptimizable(const std::string& parameter_name) override;

    virtual void clampParameters() override;

private:
    torch::Tensor _emission_tensor;

    atcg::ref_ptr<Texture2D> _density_texture;
    atcg::ref_ptr<Texture2D> _density_grad_texture;
    atcg::ref_ptr<Texture2D> _albedo_texture;
    atcg::ref_ptr<Texture2D> _albedo_grad_texture;
    int _layer_density = 0;
    int _layer_albedo  = 0;

    atcg::dref_ptr<HeterogeneousMediumData> _data_buffer;
};

}    // namespace atcg