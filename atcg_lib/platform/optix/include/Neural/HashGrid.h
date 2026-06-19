#pragma once

#include <Core/Memory.h>
#include <DataStructure/TorchUtils.h>
#include <Neural/DeviceHashGrid.h>

namespace atcg
{
template<typename T, uint32_t L, uint32_t F>
class HashGrid
{
public:
    using DeviceHashGrid_t = DeviceHashGrid<T, L, F>;

    HashGrid();

    HashGrid(const torch::Tensor& weights);

    HashGrid(const torch::Tensor& weights, uint32_t N_min, uint32_t N_max, uint32_t T);

    HashGrid(uint32_t N_min, uint32_t N_max, uint32_t T);

    DeviceHashGrid_t* getDeviceHashGrid() const { return _device_hash_grid.get(); }

    torch::Tensor getWeights() const { return _weights; }

    torch::Tensor getGradWeights() const { return _grad_weights; }

    void setWeights(const torch::Tensor& weights);

    void setGradWeights(const torch::Tensor& grad_weights);

    void zeroGradients();

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