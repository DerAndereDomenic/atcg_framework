#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>
#include <Sensor/PinholeCameraData.cuh>
#include <Sensor/SensorVPtrTable.cuh>

#include <Spectrum/SampledSpectrum.h>

extern "C" __device__ atcg::CameraRay __direct_callable__generate_ray_pinhole(const glm::vec2& raster_pos)
{
    const atcg::PinholeCameraData* pinhole_camera_data =
        *reinterpret_cast<const atcg::PinholeCameraData**>(optixGetSbtDataPointer());

    atcg::CameraRay camera_ray;

    glm::vec3 U = pinhole_camera_data->U * pinhole_camera_data->aspect_ratio;
    glm::vec3 V = pinhole_camera_data->V;
    glm::vec3 W = pinhole_camera_data->W / glm::tan(glm::radians(pinhole_camera_data->fov_y / 2.0f));

    glm::vec3 ray_dir    = glm::normalize((raster_pos.x + pinhole_camera_data->optical_center.x) * U +
                                       (raster_pos.y + pinhole_camera_data->optical_center.y) * V + W);
    glm::vec3 ray_origin = pinhole_camera_data->cam_eye;

    camera_ray.ray        = atcg::Ray(ray_origin, ray_dir);
    camera_ray.importance = atcg::SampledSpectrum(1.0f);

    return camera_ray;
}

extern "C" __device__ void __direct_callable__add_sample_pinhole(const glm::ivec3& sample_index,
                                                                 const atcg::SampledSpectrum& radiance,
                                                                 const atcg::SampledWavelengths& sampled_wavelengths)
{
    const atcg::PinholeCameraData* pinhole_camera_data =
        *reinterpret_cast<const atcg::PinholeCameraData**>(optixGetSbtDataPointer());


    glm::vec3 xyz  = radiance.toXYZ(sampled_wavelengths);
    glm::vec3 lrgb = atcg::Color::XYZ_to_lRGB(xyz);

#ifndef ATCG_SPECTRAL_RENDERING
    // Convert spectrum to lRGB
    atcg::SampledSpectrum white(1.0f);
    glm::vec3 white_lrgb = atcg::Color::XYZ_to_lRGB(white.toXYZ(sampled_wavelengths));
    // Normalize for RGB rendering
    lrgb /= white_lrgb;
#endif

    pinhole_camera_data->film->addSample(sample_index, lrgb);
}