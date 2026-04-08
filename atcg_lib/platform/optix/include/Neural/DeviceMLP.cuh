#pragma once

#include <Core/Platform.h>
#include <Core/CUDA.h>

#include <optix_types.h>
#include <cuda_fp16.h>

namespace atcg
{
template<int num_hidden, int input_size, int hidden_size, int output_size>
struct DeviceMLP
{
    CUdeviceptr _weights_buffer_ptr;
    CUdeviceptr _bias_buffer_ptr;

#ifdef __CUDACC__
    ATCG_DEVICE
    OptixCoopVec<half, output_size> forward(const OptixCoopVec<half, input_size>& input) const;
#endif
};
}    // namespace atcg

#include "../../src/Neural/DeviceMLPDetail.cuh"