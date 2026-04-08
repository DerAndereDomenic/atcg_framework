#pragma once

#include <BSDF/BSDF.h>
#include <Core/Memory.h>

#include <Neural/MLP.h>
#include <BSDF/NeuralBSDFData.cuh>

namespace atcg
{
class NeuralBSDF : public BSDF
{
public:
    NeuralBSDF(const atcg::Dictionary& dict);

    virtual ~NeuralBSDF();

    virtual void onImGuiRender() override {};

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    torch::Tensor _weights;
    DeviceBuffer<half> _weights_buffer;
    torch::Tensor _bias;

    atcg::dref_ptr<NeuralBSDFData> _bsdf_data_buffer;

    MLP<1, 8, 64, 8> _mlp;
};
}    // namespace atcg