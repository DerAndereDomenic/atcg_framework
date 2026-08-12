#pragma once

#include <Core/Optix.h>
#include <optix_types.h>
#include <cuda_fp16.h>

namespace atcg
{

enum class ActivationFunction
{
    ReLU,
    Sigmoid,
    None
};

template<ActivationFunction Func, int size>
struct Activation
{
#ifdef __CUDACC__

    static ATCG_DEVICE OptixCoopVec<half, size> forward(const OptixCoopVec<half, size>& input) { return input; }

    static ATCG_DEVICE OptixCoopVec<half, size> backward(const OptixCoopVec<half, size>& input,
                                                         const OptixCoopVec<half, size>& grad_output)
    {
        return grad_output;
    }

#endif
};

template<int size>
struct Activation<ActivationFunction::ReLU, size>
{
#ifdef __CUDACC__

    static ATCG_DEVICE OptixCoopVec<half, size> forward(const OptixCoopVec<half, size>& input)
    {
        return optixCoopVecMax(input, half(0.0f));
    }

    static ATCG_DEVICE OptixCoopVec<half, size> backward(const OptixCoopVec<half, size>& input,
                                                         const OptixCoopVec<half, size>& grad_output)
    {
        OptixCoopVec<half, size> grad_input(half(0.0f));

        // Apply ReLU backward
        for(int j = 0; j < size; ++j)
        {
            grad_input[j] = (input[j] > half(0.0f)) ? grad_output[j] : half(0.0f);
        }
        return grad_input;
    }

#endif
};

template<int size>
struct Activation<ActivationFunction::Sigmoid, size>
{
#ifdef __CUDACC__

    static ATCG_DEVICE OptixCoopVec<half, size> forward(const OptixCoopVec<half, size>& input)
    {
        const OptixCoopVec<half, size> one_vec(half(1.0f));
        const OptixCoopVec<half, size> minus_one_vec(half(-1.0f));
        const OptixCoopVec<half, size> exp = optixCoopVecExp2(optixCoopVecMul(minus_one_vec, input));
        OptixCoopVec<half, size> output    = optixCoopVecAdd(one_vec, exp);

        for(int j = 0; j < size; ++j)
        {
            output[j] = half(1.0f) / output[j];
        }
        return output;
    }

    static ATCG_DEVICE OptixCoopVec<half, size> backward(const OptixCoopVec<half, size>& input,
                                                         const OptixCoopVec<half, size>& grad_output)
    {
        const OptixCoopVec<half, size> one_vec(half(1.0f));

        OptixCoopVec<half, size> sigmoid_output           = forward(input);
        OptixCoopVec<half, size> one_minus_sigmoid_output = optixCoopVecSub(one_vec, sigmoid_output);
        OptixCoopVec<half, size> grad_input               = optixCoopVecMul(sigmoid_output, one_minus_sigmoid_output);

        return optixCoopVecMul(grad_input, grad_output);
    }

#endif
};

}    // namespace atcg