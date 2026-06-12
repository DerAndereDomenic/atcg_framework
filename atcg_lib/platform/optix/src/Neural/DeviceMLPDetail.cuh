#pragma once

#include <Neural/Linear.h>
#include <optix.h>

namespace atcg
{
#ifdef __CUDACC__

template<int num_hidden, int input_size, int hidden_size, int output_size>
ATCG_DEVICE OptixCoopVec<half, output_size> DeviceMLP<num_hidden, input_size, hidden_size, output_size>::forward(
    const OptixCoopVec<half, input_size>& input,
    OptixCoopVec<half, hidden_size>* hidden_outputs,
    OptixCoopVec<half, hidden_size>* activation_output) const
{
    using T_IN     = OptixCoopVec<half, input_size>;
    using T_OUT    = OptixCoopVec<half, output_size>;
    using T_HIDDEN = OptixCoopVec<half, hidden_size>;

    unsigned int input_layer_size = optixCoopVecGetMatrixSize<hidden_size,
                                                              input_size,
                                                              OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
                                                              OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
                                                              sizeof(half) * input_size>();

    unsigned int hidden_layer_size = optixCoopVecGetMatrixSize<hidden_size,
                                                               hidden_size,
                                                               OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
                                                               OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
                                                               sizeof(half) * hidden_size>();

    T_HIDDEN hidden   = coopVecMatMul<hidden_size, input_size>(input, _weights_buffer_ptr, 0, _bias_buffer_ptr, 0);
    hidden_outputs[0] = hidden;

    hidden               = optixCoopVecMax(hidden, half(0.0f));
    activation_output[0] = hidden;

    size_t weights_offset = input_layer_size;
    size_t bias_offset    = sizeof(half) * hidden_size;
    for(int i = 0; i < num_hidden; ++i)
    {
        hidden                = coopVecMatMul<hidden_size, hidden_size>(hidden,
                                                                        _weights_buffer_ptr,
                                                                        weights_offset,
                                                                        _bias_buffer_ptr,
                                                                        bias_offset);
        hidden_outputs[i + 1] = hidden;

        hidden                   = optixCoopVecMax(hidden, half(0.0f));
        activation_output[i + 1] = hidden;

        weights_offset += hidden_layer_size;
        bias_offset += sizeof(half) * hidden_size;
    }

    T_OUT result = coopVecMatMul<output_size, hidden_size>(hidden,
                                                           _weights_buffer_ptr,
                                                           weights_offset,
                                                           _bias_buffer_ptr,
                                                           bias_offset);
    return result;
}

template<int num_hidden, int input_size, int hidden_size, int output_size>
ATCG_DEVICE OptixCoopVec<half, input_size> DeviceMLP<num_hidden, input_size, hidden_size, output_size>::backward(
    const OptixCoopVec<half, output_size>& grad_output,
    const OptixCoopVec<half, hidden_size>* hidden_outputs,
    const OptixCoopVec<half, hidden_size>* activation_output) const
{
    using T_IN     = OptixCoopVec<half, input_size>;
    using T_OUT    = OptixCoopVec<half, output_size>;
    using T_HIDDEN = OptixCoopVec<half, hidden_size>;

    // Evaluate the forward pass again to get the intermediate activations for backward differentiation

    unsigned int input_layer_size = optixCoopVecGetMatrixSize<hidden_size,
                                                              input_size,
                                                              OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
                                                              OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
                                                              sizeof(half) * input_size>();

    unsigned int hidden_layer_size = optixCoopVecGetMatrixSize<hidden_size,
                                                               hidden_size,
                                                               OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
                                                               OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
                                                               sizeof(half) * hidden_size>();

    size_t weights_offset = input_layer_size + num_hidden * hidden_layer_size;
    size_t bias_offset    = (num_hidden + 1) * hidden_size * sizeof(half);

    // 1. Differentiate through output layer
    T_HIDDEN grad_hidden =
        coopVecMatMul<hidden_size, output_size, true>(grad_output, _weights_buffer_ptr, weights_offset);

    weights_offset -= hidden_layer_size;
    bias_offset -= sizeof(half) * hidden_size;

    // 2. Differentiate through hidden layers
    for(int i = num_hidden - 1; i >= 0; --i)
    {
        T_HIDDEN output_hidden = hidden_outputs[i + 1];
        // Apply ReLU backward
        for(int j = 0; j < hidden_size; ++j)
        {
            grad_hidden[j] = (output_hidden[j] > half(0.0f)) ? grad_hidden[j] : half(0.0f);
        }

        grad_hidden = coopVecMatMul<hidden_size, hidden_size, true>(grad_hidden, _weights_buffer_ptr, weights_offset);

        weights_offset -= hidden_layer_size;
        bias_offset -= sizeof(half) * hidden_size;
    }

    // 3. Differentiate through input layer
    T_HIDDEN output_hidden = hidden_outputs[0];
    for(int j = 0; j < hidden_size; ++j)
    {
        grad_hidden[j] = (output_hidden[j] > half(0.0f)) ? grad_hidden[j] : half(0.0f);
    }

    T_IN grad_input = coopVecMatMul<input_size, hidden_size, true>(grad_hidden, _weights_buffer_ptr, 0);

    return grad_input;
}
#endif
}    // namespace atcg