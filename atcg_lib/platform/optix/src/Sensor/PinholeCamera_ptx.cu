#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>
#include <Sensor/PinholeCameraData.cuh>
#include <Sensor/SensorVPtrTable.cuh>

#include <Spectrum/SampledSpectrum.h>

extern "C" __device__ atcg::CameraRay __direct_callable__generate_ray_pinhole(const glm::ivec2& raster_pos,
                                                                              atcg::PCG32& rng)
{
    const atcg::PinholeCameraData* pinhole_camera_data =
        *reinterpret_cast<const atcg::PinholeCameraData**>(optixGetSbtDataPointer());

    glm::vec2 jitter = rng.next2d();
    int width        = pinhole_camera_data->film->getWidth();
    int height       = pinhole_camera_data->film->getHeight();
    float u          = (((float)raster_pos.x + jitter.x) / (float)width - 0.5f) * 2.0f;
    float v          = (((float)raster_pos.y + jitter.y) / (float)height - 0.5f) * 2.0f;

    atcg::CameraRay camera_ray;

    glm::vec3 U = pinhole_camera_data->U * pinhole_camera_data->aspect_ratio;
    glm::vec3 V = pinhole_camera_data->V;
    glm::vec3 W = pinhole_camera_data->W / glm::tan(glm::radians(pinhole_camera_data->fov_y / 2.0f));

    glm::vec3 ray_dir    = glm::normalize((u + pinhole_camera_data->optical_center.x) * U +
                                          (v + pinhole_camera_data->optical_center.y) * V + W);
    glm::vec3 ray_origin = pinhole_camera_data->cam_eye;

    camera_ray.ray        = atcg::Ray(ray_origin, ray_dir);
    camera_ray.importance = atcg::SampledSpectrum(1.0f);
    camera_ray.valid      = true;

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

    pinhole_camera_data->film->addSample(sample_index, pinhole_camera_data->exposure * lrgb);
}

extern "C" __device__ atcg::PixelCoordinates __direct_callable__to_pixel_pinhole(const glm::vec3& direction)
{
    const atcg::PinholeCameraData* camera =
        *reinterpret_cast<const atcg::PinholeCameraData**>(optixGetSbtDataPointer());

    atcg::PixelCoordinates result {};
    result.valid = false;

    // Camera basis vectors, matching generate_ray_pinhole()
    glm::vec3 U = camera->U * camera->aspect_ratio;
    glm::vec3 V = camera->V;
    glm::vec3 W = camera->W / glm::tan(glm::radians(camera->fov_y / 2.0f));

    // Normalize the input direction
    glm::vec3 d = glm::normalize(direction);

    // Invert the ray direction:
    // d = normalize((u + optical_center.x) * U
    //             + (v + optical_center.y) * V + W)
    //
    // Assuming U, V, W form an orthonormal camera basis
    // before the scaling applied above, recover u and v
    // using projections onto the camera basis.

    glm::vec3 U0 = glm::normalize(camera->U);
    glm::vec3 V0 = glm::normalize(camera->V);
    glm::vec3 W0 = glm::normalize(camera->W);

    float denom = glm::dot(d, W0);
    if(denom <= 0.0f) return result;    // Behind the camera or outside the forward hemisphere

    float u = glm::dot(d, U0) / denom;
    float v = glm::dot(d, V0) / denom;

    // Undo the FOV and aspect-ratio scaling used during ray generation.
    u *= glm::length(W) / glm::length(U);
    v *= glm::length(W) / glm::length(V);

    // The original equation uses:
    // ray_dir = normalize((u + optical_center.x) * U
    //                   + (v + optical_center.y) * V + W)
    //
    // The projected coordinates above must correspond to the
    // coefficients of U and V relative to W.
    float screen_u = u - camera->optical_center.x;
    float screen_v = v - camera->optical_center.y;

    int width  = camera->film->getWidth();
    int height = camera->film->getHeight();

    glm::vec2 raster = glm::vec2((screen_u * 0.5f + 0.5f) * width, (screen_v * 0.5f + 0.5f) * height);

    result.raster_pos = glm::ivec2(static_cast<int>(glm::floor(raster.x)), static_cast<int>(glm::floor(raster.y)));

    result.subpixel_pos = glm::fract(raster);
    result.valid        = result.raster_pos.x >= 0 && result.raster_pos.x < width && result.raster_pos.y >= 0 &&
                          result.raster_pos.y < height;

    return result;
}