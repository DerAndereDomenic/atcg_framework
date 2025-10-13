#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>

#include <Math/Functions.h>
#include <Core/SurfaceInteraction.h>
#include <BSDF/BSDFVPtrTable.cuh>

extern "C" __device__ atcg::BSDFSamplingResult __direct_callable__sample_nullbsdf(const atcg::SurfaceInteraction& si,
                                                                                  atcg::PCG32& rng)
{
    atcg::BSDFSamplingResult result;

    result.bsdf_weight        = glm::vec3(1);
    result.flags              = atcg::BSDFComponentType::IdealTransmission;
    result.out_dir            = si.incoming_direction;
    result.sample_probability = 1.0f;

    return result;
}

extern "C" __device__ atcg::BSDFEvalResult __direct_callable__eval_nullbsdf(const atcg::SurfaceInteraction& si,
                                                                            const glm::vec3& outgoing_dir)
{
    atcg::BSDFEvalResult result;

    result.bsdf_value         = glm::vec3(0);
    result.sample_probability = 0.0f;
    result.flags              = atcg::BSDFComponentType::IdealTransmission;

    return result;
}