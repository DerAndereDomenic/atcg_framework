#pragma once

#include <Core/Common.h>

namespace atcg
{
template<int num_hidden, int input_size, int hidden_size, int output_size>
MLP<num_hidden, input_size, hidden_size, output_size>::MLP(const atcg::ref_ptr<RaytracingContext>& context,
                                                           const torch::Tensor& weights,
                                                           const torch::Tensor& bias)
{
    std::vector<size_t> layer_sizes;
    size_t total_size = 0;
    std::vector<OptixCoopVecMatrixDescription> in_layer_descs;
    std::vector<OptixCoopVecMatrixDescription> out_layer_descs;
    size_t offset = 0;

    for(size_t i = 0; i < num_hidden + 2; ++i)
    {
        size_t layer_size;
        OPTIX_CHECK(
            optixCoopVecMatrixComputeSize(context->getContextHandle(),
                                          i == (num_hidden + 1) ? output_size : hidden_size,
                                          i == 0 ? input_size : hidden_size,
                                          OptixCoopVecElemType::OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
                                          OptixCoopVecMatrixLayout::OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
                                          0,
                                          &layer_size));


        OptixCoopVecMatrixDescription in_desc = {};
        in_desc.N                             = i == (num_hidden + 1) ? output_size : hidden_size;
        in_desc.K                             = i == 0 ? input_size : hidden_size;
        in_desc.elementType                   = OptixCoopVecElemType::OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16;
        in_desc.layout                        = OptixCoopVecMatrixLayout::OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR;
        in_desc.rowColumnStrideInBytes        = sizeof(half) * in_desc.K;
        in_desc.sizeInBytes                   = sizeof(half) * in_desc.N * in_desc.K;
        in_desc.offsetInBytes                 = offset;

        OptixCoopVecMatrixDescription out_desc = {};
        out_desc.N                             = in_desc.N;
        out_desc.K                             = in_desc.K;
        out_desc.elementType                   = OptixCoopVecElemType::OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16;
        out_desc.layout                 = OptixCoopVecMatrixLayout::OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL;
        out_desc.rowColumnStrideInBytes = 0;    // Ignored
        out_desc.sizeInBytes            = layer_size;
        out_desc.offsetInBytes          = offset;

        in_layer_descs.push_back(in_desc);
        out_layer_descs.push_back(out_desc);

        offset += sizeof(half) * in_desc.N * in_desc.K;

        total_size += layer_size;
    }

    _weights_buffer = atcg::DeviceBuffer<half>(total_size / sizeof(half));

    OptixNetworkDescription inputNetworkDescription = {};
    inputNetworkDescription.layers                  = in_layer_descs.data();
    inputNetworkDescription.numLayers               = in_layer_descs.size();

    OptixNetworkDescription outputNetworkDescription = {};
    outputNetworkDescription.layers                  = out_layer_descs.data();
    outputNetworkDescription.numLayers               = out_layer_descs.size();

    OPTIX_CHECK(optixCoopVecMatrixConvert(context->getContextHandle(),
                                          nullptr,
                                          1,
                                          &inputNetworkDescription,
                                          (CUdeviceptr)weights.data_ptr(),
                                          0,
                                          &outputNetworkDescription,
                                          (CUdeviceptr)_weights_buffer.get(),
                                          0));

    _bias_buffer = bias;

    DeviceMLP_t device_mlp_data;
    device_mlp_data._weights_buffer_ptr = (CUdeviceptr)_weights_buffer.get();
    device_mlp_data._bias_buffer_ptr    = (CUdeviceptr)_bias_buffer.data_ptr();
    _device_mlp_buffer.upload(&device_mlp_data);
}

}    // namespace atcg