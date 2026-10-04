#pragma once

#include <Core/glm.h>
#include <Core/CUDA.h>
#include <Math/Random.h>
#include <DataStructure/Frame.h>

namespace atcg
{
struct vMFLobe
{
    glm::vec3 mu;
    float kappa;

    ATCG_INLINE ATCG_DEVICE float evalLog(const glm::vec3& w) const
    {
        const float k = kappa;

        if(k < 1e-5f)
        {
            return -glm::log(4.0f * glm::pi<float>());
        }

        const float c = k / (4.0f * glm::pi<float>() * glm::sinh(k));

        return glm::log(c) + k * glm::dot(mu, w);
    }

    ATCG_INLINE ATCG_DEVICE float eval(const glm::vec3& w) const { return glm::exp(evalLog(w)); }

    ATCG_INLINE ATCG_DEVICE glm::vec3 sample(atcg::PCG32& rng) const
    {
        float u1 = rng.nextFloat();
        float u2 = rng.nextFloat();

        float cos_theta = 1.0f + (1.0f / kappa) * glm::log(u1 + (1.0f - u1) * glm::exp(-2.0f * kappa));
        float sin_theta = glm::sqrt(1.0f - cos_theta * cos_theta);
        float phi       = 2.0f * glm::pi<float>() * u2;

        glm::vec3 w;
        w.x = sin_theta * glm::cos(phi);
        w.y = sin_theta * glm::sin(phi);
        w.z = cos_theta;

        atcg::Frame frame(mu);
        w = frame.toWorld(w);

        return w;
    }
};

struct vMFStats
{
    glm::vec3 Sk = glm::vec3(0.0f);
    float Nk     = 0.0f;
};

template<int N>
struct vMFMixture
{
    vMFLobe lobes[N];
    vMFStats stats[N];
    vMFStats batch_stats[N];

    float weights[N];

    ATCG_INLINE ATCG_DEVICE float eval(const glm::vec3& w) const
    {
        float result = 0.0f;
        for(int i = 0; i < N; ++i)
        {
            result += weights[i] * lobes[i].eval(w);
        }
        return result;
    }

    ATCG_INLINE ATCG_DEVICE glm::vec3 sample(atcg::PCG32& rng) const
    {
        float u   = rng.nextFloat();
        float sum = 0.0f;
        int index = 0;
        for(int i = 0; i < N; ++i)
        {
            sum += weights[i];
            if(u < sum)
            {
                index = i;
                break;
            }
        }
        return lobes[index].sample(rng);
    }

    ATCG_INLINE ATCG_DEVICE void accumulate(const glm::vec3& w)
    {
#ifdef __CUDACC__
        float responsibility[N];

        float responsibility_sum = 0.0f;
        for(int i = 0; i < N; ++i)
        {
            responsibility[i] = weights[i] * lobes[i].eval(w);
            responsibility_sum += responsibility[i];
        }

        for(int i = 0; i < N; ++i)
        {
            responsibility[i] /= responsibility_sum;

            atomicAdd(&batch_stats[i].Sk.x, responsibility[i] * w.x);
            atomicAdd(&batch_stats[i].Sk.y, responsibility[i] * w.y);
            atomicAdd(&batch_stats[i].Sk.z, responsibility[i] * w.z);

            atomicAdd(&batch_stats[i].Nk, responsibility[i]);
        }
        // atomicAdd(&total_mass, 1.0f);
#endif
    }

    ATCG_INLINE ATCG_DEVICE glm::vec3 mean_direction() const
    {
        glm::vec3 mean_dir = glm::vec3(0.0f);
        for(int i = 0; i < N; ++i)
        {
            mean_dir += lobes[i].mu * weights[i];
        }
        return glm::normalize(mean_dir);
    }
};
}    // namespace atcg