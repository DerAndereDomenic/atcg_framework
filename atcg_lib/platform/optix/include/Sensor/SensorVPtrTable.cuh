#pragma once

#include <DataStructure/Ray.h>
#include <Spectrum/SampledSpectrum.h>

namespace atcg
{

struct CameraRay
{
    Ray ray;
    SampledSpectrum importance;
};

struct SensorVPtrTable
{
    uint32_t generateRayCallIndex;
    uint32_t addSampleCallIndex;

#ifdef __CUDACC__
    __device__ CameraRay generateRay(const glm::vec2& raster_pos) const
    {
        return optixDirectCall<CameraRay, const glm::vec2&>(generateRayCallIndex, raster_pos);
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