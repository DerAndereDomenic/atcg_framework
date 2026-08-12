#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>
#include <Sensor/HemisphereCameraData.cuh>
#include <Sensor/SensorVPtrTable.cuh>
#include <BSDF/Sampling.h>
#include <DataStructure/Frame.h>

#include <DataStructure/SampledSpectrum.h>

extern "C" __device__ atcg::CameraRay __direct_callable__generate_ray_hemisphere(const glm::ivec2& raster_pos,
                                                                                 atcg::PCG32& rng)
{
    const atcg::HemisphereCameraData* hemisphere_camera_data =
        *reinterpret_cast<const atcg::HemisphereCameraData**>(optixGetSbtDataPointer());

    atcg::CameraRay camera_ray;

    int width        = hemisphere_camera_data->film->getWidth();
    int height       = hemisphere_camera_data->film->getHeight();
    float max_radius = (float)glm::min(width, height) / 2;

    glm::vec2 jitter = rng.next2d();
    float x          = (float)raster_pos.x - (float)width / 2.0f + jitter.x;
    float y          = (float)raster_pos.y - (float)height / 2.0f + jitter.y;

    x /= max_radius;
    y /= max_radius;

    if(x * x + y * y > 1.0f)
    {
        camera_ray.valid = false;
        return camera_ray;
    }

    glm::vec3 ray_origin = hemisphere_camera_data->cam_eye;

    atcg::Frame<glm::vec3> frame(hemisphere_camera_data->normal);

    glm::vec3 local_dir = glm::normalize(glm::vec3(x, y, glm::sqrt(glm::max(0.0f, 1.0f - x * x - y * y))));
    glm::vec3 ray_dir   = frame.toWorld(local_dir);

    camera_ray.ray        = atcg::Ray(ray_origin, ray_dir);
    camera_ray.importance = atcg::SampledSpectrum(1.0f);
    camera_ray.valid      = true;

    return camera_ray;
}

extern "C" __device__ void __direct_callable__add_sample_hemisphere(const glm::ivec3& sample_index,
                                                                    const atcg::SampledSpectrum& radiance,
                                                                    const atcg::SampledWavelengths& sampled_wavelengths)
{
    const atcg::HemisphereCameraData* hemisphere_camera_data =
        *reinterpret_cast<const atcg::HemisphereCameraData**>(optixGetSbtDataPointer());


    glm::vec3 xyz  = radiance.toXYZ(sampled_wavelengths);
    glm::vec3 lrgb = atcg::Color::XYZ_to_lRGB(xyz);

#ifndef ATCG_SPECTRAL_RENDERING
    // Convert spectrum to lRGB
    atcg::SampledSpectrum white(1.0f);
    glm::vec3 white_lrgb = atcg::Color::XYZ_to_lRGB(white.toXYZ(sampled_wavelengths));
    // Normalize for RGB rendering
    lrgb /= white_lrgb;
#endif

    hemisphere_camera_data->film->addSample(sample_index, hemisphere_camera_data->exposure * lrgb);
}