#pragma once

#include <Core/Platform.h>
#include <Core/CUDA.h>
#include <optix.h>
#include <optix_types.h>
#include <cuda_fp16.h>

namespace atcg
{
#ifdef __CUDACC__

template<int output_size,
         int input_size,
         bool transpose                  = false,
         OptixCoopVecMatrixLayout layout = OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL>
ATCG_DEVICE OptixCoopVec<half, output_size> coopVecMatMul(const OptixCoopVec<half, input_size>& input,
                                                          CUdeviceptr weights,
                                                          size_t weights_offset,
                                                          CUdeviceptr bias,
                                                          size_t bias_offset)
{
    using T_INPUT  = OptixCoopVec<half, input_size>;
    using T_OUTPUT = OptixCoopVec<half, output_size>;

    T_OUTPUT output = optixCoopVecMatMul<T_OUTPUT,
                                         T_INPUT,
                                         OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
                                         layout,
                                         transpose,
                                         output_size,
                                         input_size,
                                         OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
                                         OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16>(input,
                                                                           weights,
                                                                           weights_offset,
                                                                           bias,
                                                                           bias_offset,
                                                                           sizeof(half) * input_size);
    return output;
}

#endif
}    // namespace atcg