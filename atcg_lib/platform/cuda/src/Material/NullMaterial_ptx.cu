#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>

#include <Utils/HostDevice.h>
#include <DataStructure/SurfaceInteraction.h>
#include <Material/BSDFVPtrTable.h>

extern "C" __device__ atcg::BSDFSamplingResult
__direct_callable__sample_nullbsdf(const atcg::SurfaceInteraction& si,
                                   const atcg::SampledWavelengths& wavelengths,
                                   atcg::PCG32& rng)
{
    atcg::BSDFSamplingResult result;

    result.bsdf_weight        = atcg::SampledSpectrum(1.0f);
    result.flags              = atcg::MaterialFlag::NullTransmission;
    result.out_dir            = si.incoming_direction;
    result.sample_probability = 1.0f;

    return result;
}

extern "C" __device__ atcg::BSDFEvalResult __direct_callable__eval_nullbsdf(const atcg::SurfaceInteraction& si,
                                                                            const atcg::SampledWavelengths& wavelengths,
                                                                            const glm::vec3& outgoing_dir)
{
    atcg::BSDFEvalResult result;

    result.bsdf_value         = atcg::SampledSpectrum(0.0f);
    result.sample_probability = 0.0f;
    result.flags              = atcg::MaterialFlag::NullTransmission;

    return result;
}