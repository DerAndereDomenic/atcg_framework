#pragma once

#include <Core/glm.h>
#include <Core/SurfaceInteraction.h>
#include <Math/Random.h>
#include <BSDF/BSDFFlags.h>
#include <optix.h>

namespace atcg
{
struct BSDFSamplingResult
{
    glm::vec3 out_dir;
    glm::vec3 bsdf_weight;
    float sample_probability = 0.0f;
};

struct BSDFEvalResult
{
    glm::vec3 bsdf_value     = glm::vec3(0);
    float sample_probability = 0.0f;
};

struct BSDFVPtrTable
{
    uint32_t sampleCallIndex;
    uint32_t evalCallIndex;

    BSDFComponentType flags;

#ifdef __CUDACC__

    __device__ BSDFSamplingResult sampleBSDF(const SurfaceInteraction& si, PCG32& rng) const
    {
        return optixDirectCall<BSDFSamplingResult, const SurfaceInteraction&, PCG32&>(sampleCallIndex, si, rng);
    }

    __device__ BSDFEvalResult evalBSDF(const SurfaceInteraction& si, const glm::vec3& outgoing_dir) const
    {
        return optixDirectCall<BSDFEvalResult, const SurfaceInteraction&, const glm::vec3&>(evalCallIndex,
                                                                                            si,
                                                                                            outgoing_dir);
    }

#endif
};
}    // namespace atcg