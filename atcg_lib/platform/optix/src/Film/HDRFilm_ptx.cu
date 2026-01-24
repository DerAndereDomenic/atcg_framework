#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>
#include <Film/HDRFilmData.cuh>
#include <Film/FilmVPtrTable.cuh>

#include <Spectrum/SampledSpectrum.h>

extern "C" __device__ void __direct_callable__add_sample_hdrfilm(const glm::ivec3& sample_index,
                                                                 const glm::vec3& lrgb_radiance)
{
    const atcg::HDRFilmData* hdr_film_data = *reinterpret_cast<const atcg::HDRFilmData**>(optixGetSbtDataPointer());

    uint32_t width  = hdr_film_data->width;
    uint32_t height = hdr_film_data->height;

    uint32_t num_samples = sample_index.z;

    if(sample_index.x < 0 || sample_index.x >= (int32_t)width || sample_index.y < 0 ||
       sample_index.y >= (int32_t)height)
    {
        return;
    }

    glm::vec3* accumulation_buffer = hdr_film_data->accumulation_buffer;
    uint32_t pixel_index           = sample_index.x + width * sample_index.y;

    glm::vec3 radiance = lrgb_radiance;

    if(num_samples > 0)
    {
        // Mix with previous subframes if present!
        const float a                        = 1.0f / static_cast<float>(num_samples + 1);
        const glm::vec3 prev_output_radiance = accumulation_buffer[pixel_index];
        radiance                             = glm::lerp(prev_output_radiance, radiance, a);
    }

    accumulation_buffer[pixel_index] = radiance;
}