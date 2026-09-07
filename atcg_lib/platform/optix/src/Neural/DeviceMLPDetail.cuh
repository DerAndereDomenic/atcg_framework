#pragma once

#include <Neural/Linear.h>
#include <optix.h>

namespace atcg
{
#ifdef __CUDACC__

template<int num_hidden, int input_size, int hidden_size, int output_size, ActivationFunction activation_function>
ATCG_DEVICE OptixCoopVec<half, output_size>
DeviceMLP<num_hidden, input_size, hidden_size, output_size, activation_function>::forward(
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

    hidden               = Activation<activation_function, hidden_size>::forward(hidden);
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

        hidden                   = Activation<activation_function, hidden_size>::forward(hidden);
        activation_output[i + 1] = hidden;

        weights_offset += hidden_layer_size;
        bias_offset += sizeof(half) * hidden_size;
    }

    T_OUT result = coopVecMatMul<output_size, hidden_size>(hidden,
                                                           _weights_buffer_ptr,
                                                           weights_offset,
                                                           _bias_buffer_ptr,
                                                           bias_offset);

    // result = Activation<output_activation_function, output_size>::forward(result);
    return result;
}

template<int num_hidden, int input_size, int hidden_size, int output_size, ActivationFunction activation_function>
template<bool accumulate>
ATCG_DEVICE OptixCoopVec<half, input_size>
DeviceMLP<num_hidden, input_size, hidden_size, output_size, activation_function>::backward(
    const OptixCoopVec<half, input_size>& input,
    const OptixCoopVec<half, output_size>& grad_output,
    const OptixCoopVec<half, hidden_size>* hidden_outputs,
    const OptixCoopVec<half, hidden_size>* activation_output) const
{
    using T_IN     = OptixCoopVec<half, input_size>;
    using T_OUT    = OptixCoopVec<half, output_size>;
    using T_HIDDEN = OptixCoopVec<half, hidden_size>;

    // Evaluate the forward pass again to get the intermediate activations for backward differentiation

    unsigned int input_layer_size_forward = optixCoopVecGetMatrixSize<hidden_size,
                                                                      input_size,
                                                                      OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
                                                                      OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
                                                                      sizeof(half) * input_size>();

    unsigned int hidden_layer_size_forward = optixCoopVecGetMatrixSize<hidden_size,
                                                                       hidden_size,
                                                                       OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
                                                                       OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
                                                                       sizeof(half) * hidden_size>();

    unsigned int input_layer_size_backward = optixCoopVecGetMatrixSize<hidden_size,
                                                                       input_size,
                                                                       OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
                                                                       OPTIX_COOP_VEC_MATRIX_LAYOUT_TRAINING_OPTIMAL,
                                                                       sizeof(half) * input_size>();

    unsigned int hidden_layer_size_backward = optixCoopVecGetMatrixSize<hidden_size,
                                                                        hidden_size,
                                                                        OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
                                                                        OPTIX_COOP_VEC_MATRIX_LAYOUT_TRAINING_OPTIMAL,
                                                                        sizeof(half) * hidden_size>();

    size_t weights_offset_forward = input_layer_size_forward + num_hidden * hidden_layer_size_forward;

    size_t weights_offset_backward = input_layer_size_backward + num_hidden * hidden_layer_size_backward;
    size_t bias_offset_backward    = (num_hidden + 1) * hidden_size * sizeof(half);

    // 1. Differentiate through output layer
    if constexpr(accumulate)
    {
        optixCoopVecOuterProductAccumulate<T_OUT, T_HIDDEN>(grad_output,
                                                            activation_output[num_hidden],
                                                            _weights_gradient_buffer_ptr,
                                                            weights_offset_backward);

        optixCoopVecReduceSumAccumulate<T_OUT>(grad_output, _bias_gradient_buffer_ptr, bias_offset_backward);
    }


    T_HIDDEN grad_hidden =
        coopVecMatMul<hidden_size, output_size, true>(grad_output, _weights_buffer_ptr, weights_offset_forward);

    // 2. Differentiate through hidden layers
    for(int i = num_hidden - 1; i >= 0; --i)
    {
        weights_offset_forward -= hidden_layer_size_forward;

        weights_offset_backward -= hidden_layer_size_backward;
        bias_offset_backward -= sizeof(half) * hidden_size;

        T_HIDDEN output_hidden = hidden_outputs[i + 1];
        // Apply activation backward
        grad_hidden = Activation<activation_function, hidden_size>::backward(output_hidden, grad_hidden);

        if constexpr(accumulate)
        {
            optixCoopVecOuterProductAccumulate<T_HIDDEN, T_HIDDEN>(grad_hidden,
                                                                   activation_output[i],
                                                                   _weights_gradient_buffer_ptr,
                                                                   weights_offset_backward);

            optixCoopVecReduceSumAccumulate<T_HIDDEN>(grad_hidden, _bias_gradient_buffer_ptr, bias_offset_backward);
        }

        grad_hidden =
            coopVecMatMul<hidden_size, hidden_size, true>(grad_hidden, _weights_buffer_ptr, weights_offset_forward);
    }

    // 3. Differentiate through input layer
    T_HIDDEN output_hidden = hidden_outputs[0];
    grad_hidden            = Activation<activation_function, hidden_size>::backward(output_hidden, grad_hidden);

    if constexpr(accumulate)
    {
        optixCoopVecOuterProductAccumulate<T_HIDDEN, T_IN>(grad_hidden, input, _weights_gradient_buffer_ptr, 0);

        optixCoopVecReduceSumAccumulate<T_HIDDEN>(grad_hidden, _bias_gradient_buffer_ptr, 0);
    }

    T_IN grad_input = coopVecMatMul<input_size, hidden_size, true>(grad_hidden, _weights_buffer_ptr, 0);

    return grad_input;
}
#endif
}    // namespace atcg