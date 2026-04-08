#pragma once

#include <optix_types.h>
#include <cuda_fp16.h>

#include <Neural/DeviceMLP.cuh>


namespace atcg
{
struct NeuralBSDFData
{
    DeviceMLP<1, 8, 64, 8>* _device_mlp;
};
}    // namespace atcg