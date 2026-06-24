#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>
#include <optix.h>

#include "NeuralTextureData.cuh"

extern "C"
{
    __constant__ NeuralTextureData params;
}

using T_INPUT  = OptixCoopVec<half, 32>;
using T_OUTPUT = OptixCoopVec<half, 8>;
using T_HIDDEN = OptixCoopVec<half, 64>;

ATCG_DEVICE ATCG_INLINE T_INPUT encoding_forward(const glm::vec2& uv)
{
    // T_INPUT encoding;

    // encoding[0] = __float2half(uv.x);
    // encoding[1] = __float2half(uv.y);
    // encoding[2] = __float2half(std::sin(glm::two_pi<float>() * uv.x));
    // encoding[3] = __float2half(std::sin(glm::two_pi<float>() * uv.y));
    // encoding[4] = __float2half(std::sin(2.0f * glm::two_pi<float>() * uv.x));
    // encoding[5] = __float2half(std::sin(2.0f * glm::two_pi<float>() * uv.y));
    // encoding[6] = __float2half(std::sin(4.0f * glm::two_pi<float>() * uv.x));
    // encoding[7] = __float2half(std::sin(4.0f * glm::two_pi<float>() * uv.y));

    // return encoding;
    return params.device_hash_grid->forward(glm::vec3(uv, 0.0f));
}

extern "C" __global__ void __raygen__fwd()
{
    uint3 launch_idx = optixGetLaunchIndex();

    if(launch_idx.x >= params.width || launch_idx.y >= params.height) return;
    T_HIDDEN hidden[4];
    T_HIDDEN activations[4];

    const float u = static_cast<float>(launch_idx.x) / static_cast<float>(params.width);
    const float v = static_cast<float>(launch_idx.y) / static_cast<float>(params.height);
    const glm::vec2 uv(u, v);

    T_INPUT encoding = encoding_forward(uv);

    T_OUTPUT output = params.device_mlp->forward(encoding, hidden, activations);

    output = atcg::Activation<atcg::ActivationFunction::Sigmoid, 8>::forward(output);

    int idx = launch_idx.y * params.width + launch_idx.x;

    params.output[idx] = glm::vec4(__half2float(output[0]), __half2float(output[1]), __half2float(output[2]), 1.0f);
}

extern "C" __global__ void __raygen__bckwd()
{
    uint3 launch_idx = optixGetLaunchIndex();
    int idx          = launch_idx.y * params.width + launch_idx.x;

    if(launch_idx.x >= params.width || launch_idx.y >= params.height) return;
    T_HIDDEN hidden[4];
    T_HIDDEN activations[4];

    const float u = static_cast<float>(launch_idx.x) / static_cast<float>(params.width);
    const float v = static_cast<float>(launch_idx.y) / static_cast<float>(params.height);
    const glm::vec2 uv(u, v);

    T_INPUT encoding = encoding_forward(uv);

    T_OUTPUT output = params.device_mlp->forward(encoding, hidden, activations);

    output = atcg::Activation<atcg::ActivationFunction::Sigmoid, 8>::forward(output);

    glm::vec4 grad_output = params.grad_output[idx];

    T_OUTPUT grad_output_slice(half(0.0f));
    grad_output_slice[0] = __float2half(grad_output.x);
    grad_output_slice[1] = __float2half(grad_output.y);
    grad_output_slice[2] = __float2half(grad_output.z);

    T_OUTPUT grad_activation =
        atcg::Activation<atcg::ActivationFunction::Sigmoid, 8>::backward(output, grad_output_slice);

    T_INPUT grad_input = params.device_mlp->backward<true>(encoding, grad_activation, hidden, activations);
    params.device_hash_grid->backward<true>(glm::vec3(uv, 0.0f), grad_input);
}