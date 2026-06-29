#pragma once

#include <Core/Memory.h>
#include <DataStructure/TorchUtils.h>
#include <Neural/DeviceHashGrid.h>

namespace atcg
{
/**
 * @brief Host interface for a hash grid that can be used in optix device code.
 *
 * @tparam T The data type of the hash grid weights (e.g. half or float)
 * @tparam L The number of levels in the hash grid
 * @tparam F The number of features per hash bucket
 */
template<typename T, uint32_t L, uint32_t F>
class HashGrid
{
public:
    using DeviceHashGrid_t = DeviceHashGrid<T, L, F>;

    /**
     * @brief Constructor
     */
    HashGrid();

    /**
     * @brief Constructor
     * @note The weights should be of shape float16. If not, they are converted
     *
     * @param weights The initial weights of the hash grid (should be of shape [L, T_size, F])
     */
    HashGrid(const torch::Tensor& weights);

    /**
     * @brief Constructor
     *
     * @param weights The initial weights of the hash grid (should be of shape [L, T_size, F])
     * @param N_min The minimum resolution of the hash grid (should be a power of two)
     * @param N_max The maximum resolution of the hash grid (should be a power of two)
     * @param T_size The size of the hash table (should be a power of two)
     */
    HashGrid(const torch::Tensor& weights, uint32_t N_min, uint32_t N_max, uint32_t T);

    /**
     * @brief Constructor
     *
     * @param N_min The minimum resolution of the hash grid (should be a power of two)
     * @param N_max The maximum resolution of the hash grid (should be a power of two)
     * @param T_size The size of the hash table (should be a power of two)
     */
    HashGrid(uint32_t N_min, uint32_t N_max, uint32_t T);

    /**
     * @brief Get the device-side hash grid
     *
     * @return Pointer to the device-side hash grid
     */
    DeviceHashGrid_t* getDeviceHashGrid() const { return _device_hash_grid.get(); }

    /**
     * @brief Get the weights of the hash grid as float16 tensor
     *
     * @return The weights of the hash grid
     */
    torch::Tensor getWeights() const { return _weights; }

    /**
     * @brief Get the weight gradients of the hash grid as float tensor
     *
     * @return The weight gradients of the hash grid
     */
    torch::Tensor getGradWeights() const { return _grad_weights; }

    /**
     * @brief Set the weights of the hash grid
     * @note The weights should be of shape float16. If not, they are converted. Size should be [L, T_size, F]
     *
     * @param weights The new weights of the hash grid
     */
    void setWeights(const torch::Tensor& weights);

    /**
     * @brief Set the weight gradients of the hash grid
     * @note The weight gradients should be of shape float. Size should be [L, T_size, F]
     *
     * @param grad_weights The new weight gradients of the hash grid
     */
    void setGradWeights(const torch::Tensor& grad_weights);

    /**
     * @brief Zero the weight gradients of the hash grid
     */
    void zeroGradients();

    /**
     * @brief Upload the hash grid data to the device. This should be called after setting the weights and gradients to
     * update the device-side hash grid.
     */
    void uploadDeviceHashGridData();

private:
    void allocateBuffers();

private:
    uint32_t _N_min  = 16;
    uint32_t _N_max  = (1 << 9);
    uint32_t _T_size = (1 << 14);
    torch::Tensor _weights;
    torch::Tensor _grad_weights;    // float because optix does not support atomicAdd on half

    atcg::dref_ptr<DeviceHashGrid_t> _device_hash_grid;
};
}    // namespace atcg

#include "../../src/Neural/HashGridDetail.h"