#pragma once


#include <Core/Memory.h>
#include <Core/RaytracingContext.h>
#include <Neural/DeviceMLP.cuh>
#include <Neural/Activations.h>

#include <optix.h>


namespace atcg
{


#ifndef __CUDACC__
template<int num_hidden,
         int input_size,
         int hidden_size,
         int output_size,
         enum class ActivationFunction activation_function = ActivationFunction::ReLU>
class MLP
{
public:
    static_assert(num_hidden > 0, "Number of hidden layers must be greater than 0");
    static_assert(input_size > 0, "Input size must be greater than 0");
    static_assert(hidden_size > 0, "Hidden layer size must be greater than 0");
    static_assert(output_size > 0, "Output size must be greater than 0");
    static_assert(input_size % 8 == 0, "Input size must be a multiple of 8 for optimal memory layout");
    static_assert(hidden_size % 8 == 0, "Hidden layer size must be a multiple of 8 for optimal memory layout");
    static_assert(output_size % 8 == 0, "Output size must be a multiple of 8 for optimal memory layout");
    static_assert(input_size < 256, "Input size must be less than 256 for optimal memory layout");
    static_assert(hidden_size < 256, "Hidden layer size must be less than 256 for optimal memory layout");
    static_assert(output_size < 256, "Output size must be less than 256 for optimal memory layout");

    using DeviceMLP_t = DeviceMLP<num_hidden, input_size, hidden_size, output_size, activation_function>;

    MLP() = default;

    MLP(const atcg::ref_ptr<RaytracingContext>& context, const torch::Tensor& weights, const torch::Tensor& bias);

    void setWeights(const torch::Tensor& weights);

    void setBias(const torch::Tensor& bias);

    torch::Tensor getWeights();

    torch::Tensor getBias() const;

    void zeroGradients();

    void setWeightGradients(const torch::Tensor& weight_gradients);

    void setBiasGradients(const torch::Tensor& bias_gradients);

    torch::Tensor getWeightGradients();

    torch::Tensor getBiasGradients() const;

    void uploadDeviceMLPData();

    ATCG_INLINE DeviceMLP_t* getDeviceMLP() const { return _device_mlp_buffer.get(); }

private:
    void _initializeLayerDescriptions();

    template<OptixCoopVecMatrixLayout layout>
    size_t _computeLayerSize(int layer_idx) const;

    void _allocateBuffers();

private:
    DeviceBuffer<half> _weights_buffer;
    DeviceBuffer<half> _weights_gradient_buffer;
    torch::Tensor _bias_buffer;
    torch::Tensor _bias_gradient_buffer;

    atcg::dref_ptr<DeviceMLP_t> _device_mlp_buffer;
    atcg::ref_ptr<RaytracingContext> _context;

    std::vector<OptixCoopVecMatrixDescription> _input_layer_descs;
    size_t _input_layer_size = 0;    // in bytes
    std::vector<OptixCoopVecMatrixDescription> _output_layer_descs;
    size_t _output_layer_size = 0;    // in bytes
    std::vector<OptixCoopVecMatrixDescription> _gradient_layer_descs;
    size_t _gradient_layer_size = 0;    // in bytes
};
#endif
}    // namespace atcg

#include "../../src/Neural/MLPDetail.h"