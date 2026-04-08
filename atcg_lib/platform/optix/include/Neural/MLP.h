#pragma once


#include <Core/Memory.h>
#include <Core/RaytracingContext.h>
#include <Neural/DeviceMLP.cuh>

#include <optix.h>


namespace atcg
{


#ifndef __CUDACC__
template<int num_hidden, int input_size, int hidden_size, int output_size>
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

    using DeviceMLP_t = DeviceMLP<num_hidden, input_size, hidden_size, output_size>;

    MLP() = default;

    MLP(const atcg::ref_ptr<RaytracingContext>& context, const torch::Tensor& weights, const torch::Tensor& bias);

    ATCG_INLINE DeviceMLP_t* getDeviceMLP() const { return _device_mlp_buffer.get(); }

private:
    DeviceBuffer<half> _weights_buffer;
    torch::Tensor _bias_buffer;

    atcg::dref_ptr<DeviceMLP_t> _device_mlp_buffer;
};
#endif
}    // namespace atcg

#include "../../src/Neural/MLPDetail.h"