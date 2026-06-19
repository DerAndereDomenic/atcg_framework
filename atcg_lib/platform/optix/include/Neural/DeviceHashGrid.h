#pragma once

#include <Core/glm.h>

#include <optix.h>
#include <optix_types.h>
#include <cuda_fp16.h>

namespace atcg
{
template<typename T, uint32_t L, uint32_t F>
struct DeviceHashGrid
{
#ifdef __CUDACC__
    ATCG_DEVICE OptixCoopVec<T, L * F> forward(const glm::vec3& position);

    template<bool accumulate>
    ATCG_DEVICE glm::vec3 backward(const glm::vec3& position, const OptixCoopVec<T, L * F>& grad_output);
#endif

    uint32_t T_size;
    uint32_t N_min;
    uint32_t N_max;
    T* weights;
    float* grad_weights;
};
}    // namespace atcg

#include "../../src/Neural/DeviceHashGridDetail.h"