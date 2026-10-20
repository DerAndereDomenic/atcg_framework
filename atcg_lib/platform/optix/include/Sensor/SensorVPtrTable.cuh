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

struct PixelCoordinates
{
    glm::ivec2 raster_pos;     // integer pixel coordinatres
    glm::vec2 subpixel_pos;    // subpixel coordinates in [0, 1)^2
    bool valid = false;
};

struct SensorVPtrTable
{
    uint32_t generateRayCallIndex;
    uint32_t addSampleCallIndex;
    uint32_t toPixelCallIndex;

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

    __device__ PixelCoordinates toPixel(const glm::vec3& direction) const
    {
        return optixDirectCall<PixelCoordinates, const glm::vec3&>(toPixelCallIndex, direction);
    }
#endif
};
}    // namespace atcg