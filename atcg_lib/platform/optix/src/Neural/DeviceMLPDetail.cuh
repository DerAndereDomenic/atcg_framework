#pragma once

#include <Neural/Linear.h>

namespace atcg
{
#ifdef __CUDACC__

template<int num_hidden, int input_size, int hidden_size, int output_size>
ATCG_DEVICE OptixCoopVec<half, output_size>
DeviceMLP<num_hidden, input_size, hidden_size, output_size>::forward(const OptixCoopVec<half, input_size>& input) const
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

    T_HIDDEN hidden = coopVecMatMul<hidden_size, input_size>(input, _weights_buffer_ptr, 0, _bias_buffer_ptr, 0);

    hidden = optixCoopVecMax(hidden, half(0.0f));

    size_t weights_offset = input_layer_size;
    size_t bias_offset    = sizeof(half) * hidden_size;
    for(int i = 0; i < num_hidden; ++i)
    {
        hidden = coopVecMatMul<hidden_size, hidden_size>(hidden,
                                                         _weights_buffer_ptr,
                                                         weights_offset,
                                                         _bias_buffer_ptr,
                                                         bias_offset);

        hidden = optixCoopVecMax(hidden, half(0.0f));

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
#endif
}    // namespace atcg