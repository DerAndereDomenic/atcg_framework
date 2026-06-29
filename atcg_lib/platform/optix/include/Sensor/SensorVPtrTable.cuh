#pragma once

#include <DataStructure/Ray.h>
#include <Spectrum/SampledSpectrum.h>

namespace atcg
{

struct CameraRay
{
    Ray ray;
    SampledSpectrum importance;
    bool valid = true;
};

struct SensorVPtrTable
{
    uint32_t generateRayCallIndex;
    uint32_t addSampleCallIndex;

#ifdef __CUDACC__
    __device__ CameraRay generateRay(const glm::ivec2& raster_pos, atcg::PCG32& rng) const
    {
        return optixDirectCall<CameraRay, const glm::ivec2&, atcg::PCG32&>(generateRayCallIndex, raster_pos, rng);
    }

    __device__ void addSample(const glm::ivec3& sample_index,
                              const SampledSpectrum& radiance,
                              const SampledWavelengths& wavelengths) const
    {
        optixDirectCall<void, const glm::ivec3&, const SampledSpectrum&, const SampledWavelengths&>(addSampleCallIndex,
                                                                                                    sample_index,
                                                                                                    radiance,
                                                                                                    wavelengths);
    }
#endif
};
}    // namespace atcg