#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>

#include <Math/Functions.h>
#include <Core/SurfaceInteraction.h>
#include <BSDF/BSDFVPtrTable.cuh>

extern "C" __device__ atcg::BSDFSamplingResult
__direct_callable__sample_nullbsdf(const atcg::SurfaceInteraction& si,
                                   const atcg::SampledWavelengths& wavelengths,
                                   atcg::PCG32& rng)
{
    atcg::BSDFSamplingResult result;

    result.bsdf_weight        = atcg::SampledSpectrum(1.0f);
    result.flags              = atcg::BSDFComponentType::NullTransmission;
    result.out_dir            = si.incoming_direction;
    result.sample_probability = 1.0f;

    return result;
}

extern "C" __device__ void __direct_callable__sample_backward_nullbsdf(const atcg::SurfaceInteraction& si,
                                                                       atcg::PCG32& rng,
                                                                       const glm::vec3& dLdbsdf,
                                                                       const glm::vec3& dLdwo_)
{
    // Nothing to do since the null BSDF does not have any learnable parameters
}

extern "C" __device__ atcg::BSDFDualSamplingResult
__direct_callable__sample_forward_nullbsdf(const atcg::DualSurfaceInteraction& si,
                                           const atcg::SampledWavelengths& wavelengths,
                                           atcg::PCG32& rng)
{
    atcg::BSDFDualSamplingResult result;

    result.bsdf_weight        = CuDiff::Dual<6, glm::vec3>(glm::vec3(1.0f));
    result.flags              = atcg::BSDFComponentType::NullTransmission;
    result.out_dir            = si.incoming_direction;
    result.sample_probability = CuDiff::Dual<6, float>(1.0f);

    return result;
}


extern "C" __device__ atcg::BSDFEvalResult __direct_callable__eval_nullbsdf(const atcg::SurfaceInteraction& si,
                                                                            const atcg::SampledWavelengths& wavelengths,
                                                                            const glm::vec3& outgoing_dir)
{
    atcg::BSDFEvalResult result;

    result.bsdf_value         = atcg::SampledSpectrum(0.0f);
    result.sample_probability = 0.0f;
    result.flags              = atcg::BSDFComponentType::NullTransmission;

    return result;
}

extern "C" __device__ atcg::BSDFDualEvalResult
__direct_callable__eval_forward_nullbsdf(const atcg::DualSurfaceInteraction& si,
                                         const CuDiff::Dual<6, glm::vec3>& outgoing_dir,
                                         const atcg::SampledWavelengths& wavelengths)
{
    atcg::BSDFDualEvalResult result;

    result.bsdf_value         = CuDiff::Dual<6, glm::vec3>(glm::vec3(0.0f));
    result.sample_probability = 0.0f;
    result.flags              = atcg::BSDFComponentType::NullTransmission;

    return result;
}

extern "C" __device__ void __direct_callable__eval_backward_nullbsdf(const atcg::SurfaceInteraction& si,
                                                                     const glm::vec3& outgoing_dir,
                                                                     const glm::vec3& out_grad)
{
    // Nothing to do since the null BSDF does not have any learnable parameters
}