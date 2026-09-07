#pragma once


#include <Core/Memory.h>
#include <Core/RaytracingContext.h>
#include <Neural/DeviceMLP.cuh>
#include <Neural/Activations.h>
#include <DataStructure/TorchUtils.h>

#include <optix.h>


namespace atcg
{


#ifndef __CUDACC__

/**
 * @brief Host interface of a simple MLP using cooperative vectors
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

    /**
     * @brief Constructor
     */
    MLP() = default;

    /**
     * @brief Constructor
     * @note The weights should be float16. If not, they are converted
     *
     *
     * @param context The ray tracing context to use for cooperative vector operations
     * @param weights The initial weights of the MLP (should be of shape [input_size * hidden_size + (num_hidden - 1) *
     * hidden_size * hidden_size + hidden_size * output_size])
     * @param bias The initial bias of the MLP (should be of shape [hidden_size * num_hidden + output_size])
     */
    MLP(const atcg::ref_ptr<RaytracingContext>& context, const torch::Tensor& weights, const torch::Tensor& bias);

    /**
     * @brief Set the weights of the MLP
     * @note The weights should be float16. If not, they are converted
     *
     * @param weights The new weights of the MLP (should be of shape [input_size * hidden_size + (num_hidden - 1) *
     * hidden_size * hidden_size + hidden_size * output_size])
     */
    void setWeights(const torch::Tensor& weights);

    /**
     * @brief Set the bias of the MLP
     * @note The bias should be float16. If not, they are converted
     *
     * @param bias The new bias of the MLP (should be of shape [hidden_size * num_hidden + output_size])
     */
    void setBias(const torch::Tensor& bias);

    /**
     * @brief Get the weights of the MLP as float16 tensor
     *
     * @return The weights of the MLP
     */
    torch::Tensor getWeights();

    /**
     * @brief Get the bias of the MLP as float16 tensor
     *
     * @return The bias of the MLP
     */
    torch::Tensor getBias() const;

    /**
     * @brief Zero the gradients of the MLP
     */
    void zeroGradients();

    /**
     * @brief Set the weight gradients of the MLP
     * @note The gradients should be float16. If not, they are converted
     *
     * @param weight_gradients The new weight gradients of the MLP
     */
    void setWeightGradients(const torch::Tensor& weight_gradients);

    /**
     * @brief Set the bias gradients of the MLP
     * @note The gradients should be float16. If not, they are converted
     *
     * @param bias_gradients The new bias gradients of the MLP
     */
    void setBiasGradients(const torch::Tensor& bias_gradients);

    /**
     * @brief Get the weight gradients of the MLP as float16 tensor
     *
     * @return The weight gradients of the MLP
     */
    torch::Tensor getWeightGradients();

    /**
     * @brief Get the bias gradients of the MLP as float16 tensor
     *
     * @return The bias gradients of the MLP
     */
    torch::Tensor getBiasGradients() const;

    /**
     * @brief Upload the weights and bias to the device
     */
    void uploadDeviceMLPData();

    /**
     * @brief Get a pointer to the device MLP struct. This pointer can be passed to device code to perform forward and
     * backward passes on the MLP using cooperative vectors.
     *
     * @return A pointer to the device MLP struct
     */
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