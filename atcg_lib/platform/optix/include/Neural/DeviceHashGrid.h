#pragma once

#include <Core/glm.h>

#include <optix.h>
#include <optix_types.h>
#include <cuda_fp16.h>

namespace atcg
{
/**
 * @brief Device-side implementation of a hash grid that can be used in optix device code.
 *
 * @tparam T The data type of the hash grid weights (e.g. half or float)
 * @tparam L The number of levels in the hash grid
 * @tparam F The number of features per hash bucket
 */
template<typename T, uint32_t L, uint32_t F>
struct DeviceHashGrid
{
#ifdef __CUDACC__
    /**
     * @brief Evalute the hash grid
     *
     * @param position The 3d position
     * @return The output features of the hash grid at the given position
     */
    ATCG_DEVICE OptixCoopVec<T, L * F> forward(const glm::vec3& position);

    /**
     * @brief Backward pass for the hash grid. This function computes the gradients of the hash grid weights with
     * respect to the output gradients.
     *
     * @tparam accumulate If true, the computed gradients are accumulated to the existing gradients in the grad_weights
     * buffer
     * @param position The 3d position
     * @param grad_output The gradients of the output features
     *
     * @return The gradients of the hash grid weights at the given position
     */
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