#include "PGKernels.h"

#include <ATen/cuda/ApplyGridUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <DataStructure/TorchUtils.h>

#include <Core/CUDA.h>

namespace atcg
{
namespace detail
{
__global__ void updateMixturesKernel(vMFMixture<5>* vmf_mixtures, int N)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for(auto tid = id; tid < N; tid += num_threads)
    {
        atcg::vMFMixture<5>* vmf = &vmf_mixtures[tid];


        float total_mass = 0.0f;

        float rho = 0.5f;

        for(int i = 0; i < 5; ++i)
        {
            const float batchNk     = vmf->batch_stats[i].Nk;
            const glm::vec3 batchSk = vmf->batch_stats[i].Sk;

            // Online EM update:
            //
            // S_k <- (1-rho) S_k + rho * batchS_k
            // N_k <- (1-rho) N_k + rho * batchN_k

            vmf->stats[i].Nk = (1.0f - rho) * vmf->stats[i].Nk + rho * batchNk;

            vmf->stats[i].Sk = (1.0f - rho) * vmf->stats[i].Sk + rho * batchSk;

            total_mass += vmf->stats[i].Nk;
        }

        if(total_mass <= 1e-5f)
        {
            for(int i = 0; i < 5; ++i)
            {
                vmf->batch_stats[i].Sk = glm::vec3(0.0f);
                vmf->batch_stats[i].Nk = 0.0f;
            }

            continue;
        }

        const float inv_total_mass = 1.0f / total_mass;
        for(int i = 0; i < 5; ++i)
        {
            const float mass       = vmf->stats[i].Nk;
            const glm::vec3 sumDir = vmf->stats[i].Sk;

            const float norm_squared = glm::dot(sumDir, sumDir);
            const float norm_sum     = glm::sqrt(glm::max(1e-5f, norm_squared));

            vmf->weights[i] = mass * inv_total_mass;

            if(mass > 1e-5f)
            {
                vmf->lobes[i].mu = sumDir / norm_sum;

                // Mean resultant length.
                const float R = glm::clamp(norm_sum / mass, 0.0f, 0.9999f);

                const float R2 = R * R;

                float kappa         = R * (3.0f - R2) / (1.0f - R2);
                vmf->lobes[i].kappa = glm::clamp(kappa, 0.0f, 1e4f);
            }
            else
            {
                vmf->weights[i]     = 0.0f;
                vmf->lobes[i].kappa = 0.0f;
            }
        }

        // Normalize the mixture weights to account for floating-
        // point accumulation error and discarded components.
        float weightSum = 0.0f;

        for(int i = 0; i < 5; ++i)
        {
            weightSum += vmf->weights[i];
        }

        if(weightSum > 1e-5f)
        {
            const float invWeightSum = 1.0f / weightSum;

            for(int i = 0; i < 5; ++i)
            {
                vmf->weights[i] *= invWeightSum;
            }
        }
        else
        {
            // Defensive recovery if every component was empty.
            const float uniformWeight = 1.0f / 5.0f;

            for(int i = 0; i < 5; ++i)
            {
                vmf->weights[i]     = uniformWeight;
                vmf->lobes[i].kappa = 0.0f;
            }
        }
    }
}
}    // namespace detail

void updateMixtures(const atcg::DeviceBuffer<vMFMixture<5>>& vmf_mixtures)
{
    const int N          = vmf_mixtures.size();
    torch::Device device = atcg::GPU;

    at::cuda::CUDAGuard device_guard {device};

    const auto stream = at::cuda::getCurrentCUDAStream();

    const int threads_per_block = 128;
    const int blocks_per_grid   = (N + threads_per_block - 1) / threads_per_block;

    detail::updateMixturesKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(vmf_mixtures.get(), N);

    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
}
}    // namespace atcg