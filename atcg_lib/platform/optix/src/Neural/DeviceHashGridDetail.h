#include <cuda_fp16.h>

namespace atcg
{


#ifdef __CUDACC__

ATCG_DEVICE uint32_t _hash(const glm::ivec3& pos, uint32_t T_size)
{
    return (((uint32_t)pos.x) ^ ((uint32_t)pos.y * 2654435761) ^ ((uint32_t)pos.z * 805459861)) % T_size;
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
            float c000 = (float)layer_weights[hash000 * F + f];
            float c001 = (float)layer_weights[hash001 * F + f];
            float c010 = (float)layer_weights[hash010 * F + f];
            float c011 = (float)layer_weights[hash011 * F + f];
            float c100 = (float)layer_weights[hash100 * F + f];
            float c101 = (float)layer_weights[hash101 * F + f];
            float c110 = (float)layer_weights[hash110 * F + f];
            float c111 = (float)layer_weights[hash111 * F + f];

            float c00 = glm::mix(c000, c100, int_weights.x);
            float c01 = glm::mix(c001, c101, int_weights.x);
            float c10 = glm::mix(c010, c110, int_weights.x);
            float c11 = glm::mix(c011, c111, int_weights.x);

            float c0 = glm::mix(c00, c10, int_weights.y);
            float c1 = glm::mix(c01, c11, int_weights.y);

            result[l * F + f] = (T)glm::mix(c0, c1, int_weights.z);
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
            float g = (float)grad_output[l * F + f];

            // load corners
            float c000 = (float)layer_weights[hash000 * F + f];
            float c001 = (float)layer_weights[hash001 * F + f];
            float c010 = (float)layer_weights[hash010 * F + f];
            float c011 = (float)layer_weights[hash011 * F + f];
            float c100 = (float)layer_weights[hash100 * F + f];
            float c101 = (float)layer_weights[hash101 * F + f];
            float c110 = (float)layer_weights[hash110 * F + f];
            float c111 = (float)layer_weights[hash111 * F + f];

            // ---- forward reconstruction structure ----
            float c00 = glm::mix(c000, c100, w.x);
            float c01 = glm::mix(c001, c101, w.x);
            float c10 = glm::mix(c010, c110, w.x);
            float c11 = glm::mix(c011, c111, w.x);

            float c0 = glm::mix(c00, c10, w.y);
            float c1 = glm::mix(c01, c11, w.y);

            float c = glm::mix(c0, c1, w.z);


            // =========================
            // Backprop through mix
            // =========================

            float dc0 = g * (1.0f - w.z);
            float dc1 = g * w.z;

            float dc00 = dc0 * (1.0f - w.y);
            float dc10 = dc0 * w.y;
            float dc01 = dc1 * (1.0f - w.y);
            float dc11 = dc1 * w.y;

            float dc000 = dc00 * (1.0f - w.x);
            float dc100 = dc00 * w.x;
            float dc001 = dc01 * (1.0f - w.x);
            float dc101 = dc01 * w.x;
            float dc010 = dc10 * (1.0f - w.x);
            float dc110 = dc10 * w.x;
            float dc011 = dc11 * (1.0f - w.x);
            float dc111 = dc11 * w.x;

            if constexpr(accumulate)
            {
                atomicAdd(&layer_grads[hash000 * F + f], dc000);
                atomicAdd(&layer_grads[hash001 * F + f], dc001);
                atomicAdd(&layer_grads[hash010 * F + f], dc010);
                atomicAdd(&layer_grads[hash011 * F + f], dc011);
                atomicAdd(&layer_grads[hash100 * F + f], dc100);
                atomicAdd(&layer_grads[hash101 * F + f], dc101);
                atomicAdd(&layer_grads[hash110 * F + f], dc110);
                atomicAdd(&layer_grads[hash111 * F + f], dc111);
            }


            float dc_dw_x = 0, dc_dw_y = 0, dc_dw_z = 0;

            dc_dw_x = (1 - w.z) * ((1 - w.y) * (c100 - c000) + w.y * (c110 - c010)) +
                      w.z * ((1 - w.y) * (c101 - c001) + w.y * (c111 - c011));

            dc_dw_y = (1 - w.z) * (c10 - c00) + w.z * (c11 - c01);

            dc_dw_z = c1 - c0;

            dpos += g * glm::vec3(dc_dw_x, dc_dw_y, dc_dw_z) * (float)Nl;
        }
    }

    return dpos;
}
#endif
}    // namespace atcg