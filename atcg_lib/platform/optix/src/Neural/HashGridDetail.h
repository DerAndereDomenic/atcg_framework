namespace atcg
{
template<typename T, uint32_t L, uint32_t F>
HashGrid<T, L, F>::HashGrid()
{
    allocateBuffers();
}
template<typename T, uint32_t L, uint32_t F>
HashGrid<T, L, F>::HashGrid(const torch::Tensor& weights)
{
    allocateBuffers();
    setWeights(weights.to(torch::kFloat16));
}

template<typename T, uint32_t L, uint32_t F>
HashGrid<T, L, F>::HashGrid(const torch::Tensor& weights, uint32_t N_min, uint32_t N_max, uint32_t T)
    : _N_min(N_min),
      _N_max(N_max),
      _T_size(T)
{
    allocateBuffers();
    setWeights(weights.to(torch::kFloat16));
}

template<typename T, uint32_t L, uint32_t F>
HashGrid<T, L, F>::HashGrid(uint32_t N_min, uint32_t N_max, uint32_t T_size)
    : _N_min(N_min),
      _N_max(N_max),
      _T_size(T_size)
{
    allocateBuffers();
}

template<typename T, uint32_t L, uint32_t F>
void HashGrid<T, L, F>::allocateBuffers()
{
    // Initialize weights uniformly in [-1e-4, 1e-4]
    _weights = torch::empty({_T_size * L * F}, atcg::TensorOptions::DeviceOptions<T>());
    torch::nn::init::uniform_(_weights, -1e-4f, 1e-4f);
    _grad_weights = torch::zeros_like(_weights, atcg::TensorOptions::DeviceOptions<float>());

    uploadDeviceHashGridData();
}
template<typename T, uint32_t L, uint32_t F>
void HashGrid<T, L, F>::setWeights(const torch::Tensor& weights)
{
    _weights.copy_(weights.to(torch::kFloat16));
}

template<typename T, uint32_t L, uint32_t F>
void HashGrid<T, L, F>::setGradWeights(const torch::Tensor& grad_weights)
{
    _grad_weights.copy_(grad_weights.to(torch::kFloat));
}

template<typename T, uint32_t L, uint32_t F>
void HashGrid<T, L, F>::zeroGradients()
{
    _grad_weights.zero_();
}

template<typename T, uint32_t L, uint32_t F>
void HashGrid<T, L, F>::uploadDeviceHashGridData()
{
    DeviceHashGrid_t device_hash_grid;
    device_hash_grid.N_min        = _N_min;
    device_hash_grid.N_max        = _N_max;
    device_hash_grid.T_size       = _T_size;
    device_hash_grid.weights      = (T*)_weights.data_ptr();
    device_hash_grid.grad_weights = (float*)_grad_weights.data_ptr();

    _device_hash_grid.upload(&device_hash_grid);
}

}    // namespace atcg