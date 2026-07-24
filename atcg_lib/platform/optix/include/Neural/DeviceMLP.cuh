#pragma once

#include <Core/Platform.h>
#include <Core/CUDA.h>
#include <Neural/Activations.h>

#include <optix_types.h>
#include <cuda_fp16.h>

namespace atcg
{
/**
 * @brief A device-side MLP using cooperative vectors.
 *
 * @tparam num_hidden Number of hidden layers
 * @tparam input_size Size of the input layer (must be a multiple of 8 and less than 256 for optimal memory layout)
 * @tparam hidden_size Size of the hidden layers (must be a multiple of 8 and less than 256 for optimal memory layout)
 * @tparam output_size Size of the output layer (must be a multiple of 8 and less than 256 for optimal memory layout)
 * @tparam activation_function Activation function to use (default: ReLU)
 */
template<int num_hidden,
         int input_size,
         int hidden_size,
         int output_size,
         ActivationFunction activation_function = ActivationFunction::ReLU>
struct DeviceMLP
{
    CUdeviceptr _weights_buffer_ptr;
    CUdeviceptr _bias_buffer_ptr;
    CUdeviceptr _weights_gradient_buffer_ptr;
    CUdeviceptr _bias_gradient_buffer_ptr;

#ifdef __CUDACC__
    /**
     * @brief Perform a forward pass through the MLP using cooperative vectors. The input and output are expected to be
     * in half precision.
     *
     * @param input Cooperative vector containing the input to the MLP. Must be of size `input_size`.
     * @param hidden_outputs Array of cooperative vectors to store the outputs of the hidden layers. (should have
     * `num_hidden` +1 elements, each of size `hidden_size`)
     * @param activation_output Array of cooperative vectors to store the outputs of the activation functions applied to
     * the hidden layers. (should have `num_hidden` elements, each of size `hidden_size`)
     *
     * @return Cooperative vector containing the output of the MLP
     */
    ATCG_DEVICE
    OptixCoopVec<half, output_size> forward(const OptixCoopVec<half, input_size>& input,
                                            OptixCoopVec<half, hidden_size>* hidden_outputs,
                                            OptixCoopVec<half, hidden_size>* activation_output) const;

    /**
     * @brief Perform a backward pass through the MLP using cooperative vectors. The input and output gradients are
     * expected to be in half precision.
     *
     * @tparam accumulate If true, the computed gradients will be accumulated into the existing values in the gradient
     * buffers
     * @param input Cooperative vector containing the input to the MLP. Must be of size `input_size`.
     * @param grad_output Cooperative vector containing the gradient of the loss with respect to the output of the MLP.
     * Must be of size `output_size`.
     * @param hidden_outputs Array of cooperative vectors containing the outputs of the hidden layers from the forward
     * pass. (should have `num_hidden` +1 elements, each of size `hidden_size`)
     * @param activation_output Array of cooperative vectors containing the outputs of the activation functions applied
     * to the hidden layers from the forward pass. (should have `num_hidden` elements, each of size `hidden_size`)
     *
     * @return Cooperative vector containing the gradient of the loss with respect to the input of the MLP. This is
     * useful for chaining multiple MLPs together or for computing gradients with respect to the input features.
     */
    template<bool accumulate = false>
    ATCG_DEVICE OptixCoopVec<half, input_size> backward(const OptixCoopVec<half, input_size>& input,
                                                        const OptixCoopVec<half, output_size>& grad_output,
                                                        const OptixCoopVec<half, hidden_size>* hidden_outputs,
                                                        const OptixCoopVec<half, hidden_size>* activation_output) const;
#endif
};
}    // namespace atcg

#include "../../src/Neural/DeviceMLPDetail.cuh"