#pragma once

#include <Neural/MLP.h>
#include <Neural/HashGrid.h>
#include <Core/RaytracingPipeline.h>
#include <Core/ShaderBindingTable.h>
#include "NeuralTextureData.cuh"

#include <torch/torch.h>

class NeuralTexture;

struct NeuralTextureNode : public torch::autograd::Node
{
    NeuralTexture* neural_texture;

    torch::autograd::variable_list apply(torch::autograd::variable_list&& grads) override;

    virtual void release_variables() override;
};

class NeuralTexture
{
public:
    NeuralTexture(const atcg::ref_ptr<atcg::RaytracingContext>& context);

    ~NeuralTexture();

    torch::Tensor evaluate(const uint32_t width, const uint32_t height);

    ATCG_INLINE std::vector<torch::Tensor> getParameters() const { return {_weights, _bias, _hash_weights}; }

    atcg::MLP<3, 32, 64, 8>& getMLP() { return _mlp; }

private:
    friend class NeuralTextureNode;

    torch::Tensor _forward(const uint32_t width, const uint32_t height);
    std::vector<torch::Tensor> _backward(const torch::Tensor& grad_output);

private:
    atcg::MLP<3, 32, 64, 8> _mlp;
    atcg::HashGrid<half, 16, 2> _hash_grid;

    torch::Tensor _weights;
    torch::Tensor _bias;
    torch::Tensor _hash_weights;

    uint32_t _fwd_call_index;
    uint32_t _bckwd_call_index;
    atcg::ref_ptr<atcg::RayTracingPipeline> _pipeline;
    atcg::ref_ptr<atcg::ShaderBindingTable> _sbt;

    atcg::dref_ptr<NeuralTextureData> _launch_params;

    bool _weights_dirty = false;
};