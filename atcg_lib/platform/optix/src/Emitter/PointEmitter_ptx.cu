#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>
#include <Math/Functions.h>
#include <Core/SurfaceInteraction.h>

#include <Emitter/EmitterVPtrTable.cuh>
#include <Emitter/PointEmitterData.cuh>


extern "C" __device__ atcg::EmitterSamplingResult
__direct_callable__sample_pointemitter(const atcg::SurfaceInteraction& si, atcg::PCG32& rng)
{
    const atcg::PointEmitterData* sbt_data =
        *reinterpret_cast<const atcg::PointEmitterData**>(optixGetSbtDataPointer());

    atcg::EmitterSamplingResult result;

    glm::vec3 dir_to_light = sbt_data->position - si.position;
    float distance         = glm::length(dir_to_light);

    result.direction_to_light          = dir_to_light / (1e-5f + distance);
    result.distance_to_light           = distance;
    result.radiance_weight_at_receiver = sbt_data->color * sbt_data->intensity / (distance * distance);
    result.sampling_pdf                = 1.0f;

    return result;
}

extern "C" __device__ glm::vec3 __direct_callable__eval_pointemitter(const atcg::SurfaceInteraction& si)
{
    const atcg::PointEmitterData* sbt_data =
        *reinterpret_cast<const atcg::PointEmitterData**>(optixGetSbtDataPointer());

    return sbt_data->color * sbt_data->intensity;
}

extern "C" __device__ float __direct_callable__evalpdf_pointemitter(const atcg::SurfaceInteraction& last_si,
                                                                    const atcg::SurfaceInteraction& si)
{
    const atcg::PointEmitterData* sbt_data =
        *reinterpret_cast<const atcg::PointEmitterData**>(optixGetSbtDataPointer());

    return 0.0f;
}