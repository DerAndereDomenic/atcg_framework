#include <cuda_fp16.h>

namespace atcg
{


#ifdef __CUDACC__

ATCG_DEVICE uint32_t _hash(const glm::ivec3& pos, uint32_t T_size)
{
    return ((pos.x) ^ (pos.y * 2654435761) ^ (pos.z * 805459861)) % T_size;
}

template<typename T>
ATCG_DEVICE T _mix(const T& a, const T& b, const T& t)
{
    return a * (T(1.0f) - t) + b * t;
}

template<typename T, uint32_t L, uint32_t F>
ATCG_DEVICE OptixCoopVec<T, L * F> DeviceHashGrid<T, L, F>::forward(const glm::vec3& position)
{
    using T_OUT = OptixCoopVec<T, L * F>;

    T_OUT result;

    float b = glm::exp((glm::log((float)N_max) - glm::log((float)N_min)) / (float)(L - 1));

    for(int l = 0; l < L; ++l)
    {
        T* layer_weights     = weights + (l * T_size * F);
        uint32_t Nl          = (uint32_t)(N_min * glm::pow(b, (float)l));
        glm::vec3 pos_scaled = position * (float)Nl;
        glm::ivec3 pos_floor = glm::floor(pos_scaled);

        uint32_t hash000 = _hash(pos_floor + glm::ivec3(0, 0, 0), T_size);
        uint32_t hash001 = _hash(pos_floor + glm::ivec3(0, 0, 1), T_size);
        uint32_t hash010 = _hash(pos_floor + glm::ivec3(0, 1, 0), T_size);
        uint32_t hash011 = _hash(pos_floor + glm::ivec3(0, 1, 1), T_size);
        uint32_t hash100 = _hash(pos_floor + glm::ivec3(1, 0, 0), T_size);
        uint32_t hash101 = _hash(pos_floor + glm::ivec3(1, 0, 1), T_size);
        uint32_t hash110 = _hash(pos_floor + glm::ivec3(1, 1, 0), T_size);
        uint32_t hash111 = _hash(pos_floor + glm::ivec3(1, 1, 1), T_size);

        glm::vec3 int_weights = pos_scaled - glm::vec3(pos_floor);

        // Interpolate
        for(int f = 0; f < F; ++f)
        {
            T c000 = layer_weights[hash000 * F + f];
            T c001 = layer_weights[hash001 * F + f];
            T c010 = layer_weights[hash010 * F + f];
            T c011 = layer_weights[hash011 * F + f];
            T c100 = layer_weights[hash100 * F + f];
            T c101 = layer_weights[hash101 * F + f];
            T c110 = layer_weights[hash110 * F + f];
            T c111 = layer_weights[hash111 * F + f];

            T c00 = _mix(c000, c100, (T)int_weights.x);
            T c01 = _mix(c001, c101, (T)int_weights.x);
            T c10 = _mix(c010, c110, (T)int_weights.x);
            T c11 = _mix(c011, c111, (T)int_weights.x);

            T c0 = _mix(c00, c10, (T)int_weights.y);
            T c1 = _mix(c01, c11, (T)int_weights.y);

            result[l * F + f] = _mix(c0, c1, (T)int_weights.z);
        }
    }

    return result;
}

template<typename T, uint32_t L, uint32_t F>
template<bool accumulate>
ATCG_DEVICE glm::vec3 DeviceHashGrid<T, L, F>::backward(const glm::vec3& position,
                                                        const OptixCoopVec<T, L * F>& grad_output)
{
    float b = glm::exp((glm::log((float)N_max) - glm::log((float)N_min)) / (float)(L - 1));

    glm::vec3 dpos = glm::vec3(0.0f);

    for(int l = 0; l < L; ++l)
    {
        T* layer_weights   = weights + (l * T_size * F);
        float* layer_grads = grad_weights + (l * T_size * F);

        uint32_t Nl = (uint32_t)(N_min * glm::pow(b, (float)l));

        glm::vec3 pos_scaled = position * (float)Nl;
        glm::ivec3 pos_floor = glm::floor(pos_scaled);

        glm::vec3 w = pos_scaled - glm::vec3(pos_floor);

        uint32_t hash000 = _hash(pos_floor + glm::ivec3(0, 0, 0), T_size);
        uint32_t hash001 = _hash(pos_floor + glm::ivec3(0, 0, 1), T_size);
        uint32_t hash010 = _hash(pos_floor + glm::ivec3(0, 1, 0), T_size);
        uint32_t hash011 = _hash(pos_floor + glm::ivec3(0, 1, 1), T_size);
        uint32_t hash100 = _hash(pos_floor + glm::ivec3(1, 0, 0), T_size);
        uint32_t hash101 = _hash(pos_floor + glm::ivec3(1, 0, 1), T_size);
        uint32_t hash110 = _hash(pos_floor + glm::ivec3(1, 1, 0), T_size);
        uint32_t hash111 = _hash(pos_floor + glm::ivec3(1, 1, 1), T_size);

        for(int f = 0; f < F; ++f)
        {
            T g = grad_output[l * F + f];

            // load corners
            T c000 = layer_weights[hash000 * F + f];
            T c001 = layer_weights[hash001 * F + f];
            T c010 = layer_weights[hash010 * F + f];
            T c011 = layer_weights[hash011 * F + f];
            T c100 = layer_weights[hash100 * F + f];
            T c101 = layer_weights[hash101 * F + f];
            T c110 = layer_weights[hash110 * F + f];
            T c111 = layer_weights[hash111 * F + f];

            // ---- forward reconstruction structure ----
            T c00 = _mix(c000, c100, (T)w.x);
            T c01 = _mix(c001, c101, (T)w.x);
            T c10 = _mix(c010, c110, (T)w.x);
            T c11 = _mix(c011, c111, (T)w.x);

            T c0 = _mix(c00, c10, (T)w.y);
            T c1 = _mix(c01, c11, (T)w.y);

            T c = _mix(c0, c1, (T)w.z);


            // =========================
            // Backprop through mix
            // =========================

            T dc0 = g * (T(1.0f) - (T)w.z);
            T dc1 = g * (T)w.z;

            T dc00 = dc0 * (T(1.0f) - (T)w.y);
            T dc10 = dc0 * (T)w.y;
            T dc01 = dc1 * (T(1.0f) - (T)w.y);
            T dc11 = dc1 * (T)w.y;

            T dc000 = dc00 * (T(1.0f) - (T)w.x);
            T dc100 = dc00 * (T)w.x;
            T dc001 = dc01 * (T(1.0f) - (T)w.x);
            T dc101 = dc01 * (T)w.x;
            T dc010 = dc10 * (T(1.0f) - (T)w.x);
            T dc110 = dc10 * (T)w.x;
            T dc011 = dc11 * (T(1.0f) - (T)w.x);
            T dc111 = dc11 * (T)w.x;

            if constexpr(accumulate)
            {
                atomicAdd(&layer_grads[hash000 * F + f], (float)dc000);
                atomicAdd(&layer_grads[hash001 * F + f], (float)dc001);
                atomicAdd(&layer_grads[hash010 * F + f], (float)dc010);
                atomicAdd(&layer_grads[hash011 * F + f], (float)dc011);
                atomicAdd(&layer_grads[hash100 * F + f], (float)dc100);
                atomicAdd(&layer_grads[hash101 * F + f], (float)dc101);
                atomicAdd(&layer_grads[hash110 * F + f], (float)dc110);
                atomicAdd(&layer_grads[hash111 * F + f], (float)dc111);
            }

            glm::vec3 dpos_scaled(0.0f);

            T dwdx = 0, dwdy = 0, dwdz = 0;

            // x
            dwdx += (dc100 + dc101 + dc110 + dc111) - (dc000 + dc001 + dc010 + dc011);
            // y
            dwdy += (dc010 + dc011 + dc110 + dc111) - (dc000 + dc001 + dc100 + dc101);
            // z
            dwdz += (dc001 + dc011 + dc101 + dc111) - (dc000 + dc010 + dc100 + dc110);

            dpos_scaled += glm::vec3(dwdx, dwdy, dwdz);

            dpos += dpos_scaled * (float)Nl;
        }
    }

    return dpos;
}
#endif
}    // namespace atcg