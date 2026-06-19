#pragma once

#include <Core/Common.h>
#include <Utils/Utils.h>
#include <Neural/CoopVec.h>

namespace atcg
{
template<int num_hidden,
         int input_size,
         int hidden_size,
         int output_size,
         enum class ActivationFunction activation_function>
MLP<num_hidden, input_size, hidden_size, output_size, activation_function>::MLP(
    const atcg::ref_ptr<RaytracingContext>& context,
    const torch::Tensor& weights,
    const torch::Tensor& bias)
    : _context(context)
{
    _initializeLayerDescriptions();
    _allocateBuffers();
    _bias_buffer          = torch::zeros_like(bias);
    _bias_gradient_buffer = torch::zeros_like(bias);

    setWeights(weights);
    setBias(bias);
    zeroGradients();

    uploadDeviceMLPData();
}

template<int num_hidden,
         int input_size,
         int hidden_size,
         int output_size,
         enum class ActivationFunction activation_function>
void MLP<num_hidden, input_size, hidden_size, output_size, activation_function>::setWeights(
    const torch::Tensor& weights)
{
    OptixNetworkDescription inputNetworkDescription = {};
    inputNetworkDescription.layers                  = _input_layer_descs.data();
    inputNetworkDescription.numLayers               = _input_layer_descs.size();

    OptixNetworkDescription outputNetworkDescription = {};
    outputNetworkDescription.layers                  = _output_layer_descs.data();
    outputNetworkDescription.numLayers               = _output_layer_descs.size();

    atcg::CoopVecMatrixConvert(_context->getContextHandle(),
                               nullptr,
                               1,
                               &inputNetworkDescription,
                               (CUdeviceptr)weights.data_ptr(),
                               0,
                               &outputNetworkDescription,
                               (CUdeviceptr)_weights_buffer.get(),
                               0);
}

template<int num_hidden,
         int input_size,
         int hidden_size,
         int output_size,
         enum class ActivationFunction activation_function>
void MLP<num_hidden, input_size, hidden_size, output_size, activation_function>::setBias(const torch::Tensor& bias)
{
    _bias_buffer.copy_(bias);
}

template<int num_hidden,
         int input_size,
         int hidden_size,
         int output_size,
         enum class ActivationFunction activation_function>
torch::Tensor MLP<num_hidden, input_size, hidden_size, output_size, activation_function>::getWeights()
{
    torch::Tensor weights =
        torch::empty({(int)(_input_layer_size / sizeof(half))}, atcg::TensorOptions::halfDeviceOptions());

    OptixNetworkDescription outputNetworkDescription = {};
    outputNetworkDescription.layers                  = _input_layer_descs.data();
    outputNetworkDescription.numLayers               = _input_layer_descs.size();

    OptixNetworkDescription inputNetworkDescription = {};
    inputNetworkDescription.layers                  = _output_layer_descs.data();
    inputNetworkDescription.numLayers               = _output_layer_descs.size();

    atcg::CoopVecMatrixConvert(_context->getContextHandle(),
                               nullptr,
                               1,
                               &inputNetworkDescription,
                               (CUdeviceptr)_weights_buffer.get(),
                               0,
                               &outputNetworkDescription,
                               (CUdeviceptr)weights.data_ptr(),
                               0);

    return weights;
}

template<int num_hidden,
         int input_size,
         int hidden_size,
         int output_size,
         enum class ActivationFunction activation_function>
torch::Tensor MLP<num_hidden, input_size, hidden_size, output_size, activation_function>::getBias() const
{
    return _bias_buffer;
}

template<int num_hidden,
         int input_size,
         int hidden_size,
         int output_size,
         enum class ActivationFunction activation_function>
void MLP<num_hidden, input_size, hidden_size, output_size, activation_function>::zeroGradients()
{
    setWeightGradients(
        torch::zeros({(int)(_input_layer_size / sizeof(half))}, atcg::TensorOptions::halfDeviceOptions()));
    setBiasGradients(torch::zeros_like(_bias_buffer));
}

template<int num_hidden,
         int input_size,
         int hidden_size,
         int output_size,
         enum class ActivationFunction activation_function>
void MLP<num_hidden, input_size, hidden_size, output_size, activation_function>::setWeightGradients(
    const torch::Tensor& weights_gradients)
{
    OptixNetworkDescription inputNetworkDescription = {};
    inputNetworkDescription.layers                  = _input_layer_descs.data();
    inputNetworkDescription.numLayers               = _input_layer_descs.size();

    OptixNetworkDescription outputNetworkDescription = {};
    outputNetworkDescription.layers                  = _gradient_layer_descs.data();
    outputNetworkDescription.numLayers               = _gradient_layer_descs.size();

    atcg::CoopVecMatrixConvert(_context->getContextHandle(),
                               nullptr,
                               1,
                               &inputNetworkDescription,
                               (CUdeviceptr)weights_gradients.data_ptr(),
                               0,
                               &outputNetworkDescription,
                               (CUdeviceptr)_weights_gradient_buffer.get(),
                               0);
}

template<int num_hidden,
         int input_size,
         int hidden_size,
         int output_size,
         enum class ActivationFunction activation_function>
void MLP<num_hidden, input_size, hidden_size, output_size, activation_function>::setBiasGradients(
    const torch::Tensor& bias_gradients)
{
    _bias_gradient_buffer.copy_(bias_gradients);
}

template<int num_hidden,
         int input_size,
         int hidden_size,
         int output_size,
         enum class ActivationFunction activation_function>
torch::Tensor MLP<num_hidden, input_size, hidden_size, output_size, activation_function>::getWeightGradients()
{
    torch::Tensor gradients =
        torch::zeros({(int)(_input_layer_size / sizeof(half))}, atcg::TensorOptions::halfDeviceOptions());

    OptixNetworkDescription outputNetworkDescription = {};
    outputNetworkDescription.layers                  = _input_layer_descs.data();
    outputNetworkDescription.numLayers               = _input_layer_descs.size();

    OptixNetworkDescription inputNetworkDescription = {};
    inputNetworkDescription.layers                  = _gradient_layer_descs.data();
    inputNetworkDescription.numLayers               = _gradient_layer_descs.size();

    atcg::CoopVecMatrixConvert(_context->getContextHandle(),
                               nullptr,
                               1,
                               &inputNetworkDescription,
                               (CUdeviceptr)_weights_gradient_buffer.get(),
                               0,
                               &outputNetworkDescription,
                               (CUdeviceptr)gradients.data_ptr(),
                               0);

    return gradients;
}

template<int num_hidden,
         int input_size,
         int hidden_size,
         int output_size,
         enum class ActivationFunction activation_function>
torch::Tensor MLP<num_hidden, input_size, hidden_size, output_size, activation_function>::getBiasGradients() const
{
    return _bias_gradient_buffer;
}

template<int num_hidden,
         int input_size,
         int hidden_size,
         int output_size,
         enum class ActivationFunction activation_function>
void MLP<num_hidden, input_size, hidden_size, output_size, activation_function>::uploadDeviceMLPData()
{
    DeviceMLP_t device_mlp_data;
    device_mlp_data._weights_buffer_ptr          = (CUdeviceptr)_weights_buffer.get();
    device_mlp_data._bias_buffer_ptr             = (CUdeviceptr)_bias_buffer.data_ptr();
    device_mlp_data._weights_gradient_buffer_ptr = (CUdeviceptr)_weights_gradient_buffer.get();
    device_mlp_data._bias_gradient_buffer_ptr    = (CUdeviceptr)_bias_gradient_buffer.data_ptr();
    _device_mlp_buffer.upload(&device_mlp_data);
}

template<int num_hidden,
         int input_size,
         int hidden_size,
         int output_size,
         enum class ActivationFunction activation_function>
void MLP<num_hidden, input_size, hidden_size, output_size, activation_function>::_initializeLayerDescriptions()
{
    for(size_t i = 0; i < num_hidden + 2; ++i)
    {
        size_t output_layer_size =
            _computeLayerSize<OptixCoopVecMatrixLayout::OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL>(i);

        size_t gradient_layer_size =
            _computeLayerSize<OptixCoopVecMatrixLayout::OPTIX_COOP_VEC_MATRIX_LAYOUT_TRAINING_OPTIMAL>(i);

        OptixCoopVecMatrixDescription in_desc = {};
        in_desc.N                             = i == (num_hidden + 1) ? output_size : hidden_size;
        in_desc.K                             = i == 0 ? input_size : hidden_size;
        in_desc.elementType                   = OptixCoopVecElemType::OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16;
        in_desc.layout                        = OptixCoopVecMatrixLayout::OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR;
        in_desc.rowColumnStrideInBytes        = sizeof(half) * in_desc.K;
        in_desc.sizeInBytes                   = sizeof(half) * in_desc.N * in_desc.K;
        in_desc.offsetInBytes                 = _input_layer_size;

        OptixCoopVecMatrixDescription out_desc = {};
        out_desc.N                             = in_desc.N;
        out_desc.K                             = in_desc.K;
        out_desc.elementType                   = OptixCoopVecElemType::OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16;
        out_desc.layout                 = OptixCoopVecMatrixLayout::OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL;
        out_desc.rowColumnStrideInBytes = 0;    // Ignored
        out_desc.sizeInBytes            = output_layer_size;
        out_desc.offsetInBytes          = _output_layer_size;

        OptixCoopVecMatrixDescription gradient_desc = {};
        gradient_desc.N                             = in_desc.N;
        gradient_desc.K                             = in_desc.K;
        gradient_desc.elementType                   = OptixCoopVecElemType::OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16;
        gradient_desc.layout                 = OptixCoopVecMatrixLayout::OPTIX_COOP_VEC_MATRIX_LAYOUT_TRAINING_OPTIMAL;
        gradient_desc.rowColumnStrideInBytes = 0;    // Ignored
        gradient_desc.sizeInBytes            = gradient_layer_size;
        gradient_desc.offsetInBytes          = _gradient_layer_size;

        _input_layer_descs.push_back(in_desc);
        _output_layer_descs.push_back(out_desc);
        _gradient_layer_descs.push_back(gradient_desc);

        _input_layer_size += sizeof(half) * in_desc.N * in_desc.K;
        _output_layer_size += output_layer_size;
        _gradient_layer_size += gradient_layer_size;
    }
}

template<int num_hidden,
         int input_size,
         int hidden_size,
         int output_size,
         enum class ActivationFunction activation_function>
template<OptixCoopVecMatrixLayout layout>
size_t
MLP<num_hidden, input_size, hidden_size, output_size, activation_function>::_computeLayerSize(int layer_idx) const
{
    size_t layer_size;
    atcg::CoopVecMatrixComputeSize(_context->getContextHandle(),
                                   layer_idx == (num_hidden + 1) ? output_size : hidden_size,
                                   layer_idx == 0 ? input_size : hidden_size,
                                   OptixCoopVecElemType::OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
                                   layout,
                                   0,
                                   &layer_size);
    return layer_size;
}

template<int num_hidden,
         int input_size,
         int hidden_size,
         int output_size,
         enum class ActivationFunction activation_function>
void MLP<num_hidden, input_size, hidden_size, output_size, activation_function>::_allocateBuffers()
{
    _weights_buffer          = atcg::DeviceBuffer<half>(_output_layer_size / sizeof(half));
    _weights_gradient_buffer = atcg::DeviceBuffer<half>(_gradient_layer_size / sizeof(half));
}

}    // namespace atcg