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

#ifdef __CUDACC__

    __device__ void addSample(const glm::ivec3& sample_index, const glm::vec3& lrgb_radiance) const
    {
        optixDirectCall<void, const glm::ivec3&, const glm::vec3&>(addSampleCallIndex, sample_index, lrgb_radiance);
    }
#endif
};
}    // namespace atcg