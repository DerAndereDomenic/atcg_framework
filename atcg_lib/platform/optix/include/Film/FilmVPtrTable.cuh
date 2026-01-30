#pragma once

#include <Core/glm.h>
#include <Spectrum/SampledSpectrum.h>
#include <Math/Random.h>
#include <optix.h>

namespace atcg
{
struct FilmVPtrTable
{
    uint32_t addSampleCallIndex;
    uint32_t getWidthCallIndex;
    uint32_t getHeightCallIndex;

#ifdef __CUDACC__

    __device__ void addSample(const glm::ivec3& sample_index, const glm::vec3& lrgb_radiance) const
    {
        optixDirectCall<void, const glm::ivec3&, const glm::vec3&>(addSampleCallIndex, sample_index, lrgb_radiance);
    }

    __device__ uint32_t getWidth() const { return optixDirectCall<uint32_t>(getWidthCallIndex); }

    __device__ uint32_t getHeight() const { return optixDirectCall<uint32_t>(getHeightCallIndex); }
#endif
};
}    // namespace atcg