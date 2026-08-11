#pragma once

#include <Core/glm.h>
#include <DataStructure/SurfaceInteraction.h>
#include <Math/Random.h>
#include <BSDF/BSDFFlags.h>
#include <Spectrum/SampledSpectrum.h>
#include <optix.h>


namespace atcg
{
struct BSDFSamplingResult
{
    glm::vec3 out_dir;
    SampledSpectrum bsdf_weight;
    float sample_probability = 0.0f;
    BSDFComponentType flags  = BSDFComponentType::Any;
};

struct BSDFEvalResult
{
    SampledSpectrum bsdf_value = SampledSpectrum(0);
    float sample_probability   = 0.0f;
    BSDFComponentType flags    = BSDFComponentType::Any;
};

struct BSDFVPtrTable
{
    uint32_t sampleCallIndex;
    uint32_t evalCallIndex;

    BSDFComponentType flags;

#ifdef __CUDACC__

    __device__ BSDFSamplingResult sampleBSDF(const SurfaceInteraction& si,
                                             const atcg::SampledWavelengths& wavelengths,
                                             PCG32& rng) const
    {
        return optixDirectCall<BSDFSamplingResult, const SurfaceInteraction&, const atcg::SampledWavelengths&, PCG32&>(
            sampleCallIndex,
            si,
            wavelengths,
            rng);
    }

    __device__ BSDFEvalResult evalBSDF(const SurfaceInteraction& si,
                                       const glm::vec3& outgoing_dir,
                                       const atcg::SampledWavelengths& wavelengths) const
    {
        return optixDirectCall<BSDFEvalResult,
                               const SurfaceInteraction&,
                               const glm::vec3&,
                               const atcg::SampledWavelengths&>(evalCallIndex, si, outgoing_dir, wavelengths);
    }

#endif
};

}    // namespace atcg