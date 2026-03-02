#pragma once

#include <Core/glm.h>
#include <Core/SurfaceInteraction.h>
#include <Math/Random.h>
#include <BSDF/BSDFFlags.h>
#include <Spectrum/SampledSpectrum.h>
#include <optix.h>

#include <CuDiff/Dual.h>

namespace atcg
{
struct BSDFSamplingResult
{
    glm::vec3 out_dir;
    SampledSpectrum bsdf_weight;
    float sample_probability = 0.0f;
    BSDFComponentType flags  = BSDFComponentType::Any;
};

struct BSDFDualSamplingResult
{
    CuDiff::Dual<6, glm::vec3> out_dir;
    CuDiff::Dual<6, glm::vec3> bsdf_weight;
    CuDiff::Dual<6, float> sample_probability;
    BSDFComponentType flags = BSDFComponentType::Any;
};

struct BSDFEvalResult
{
    SampledSpectrum bsdf_value = SampledSpectrum(0);
    float sample_probability   = 0.0f;
    BSDFComponentType flags    = BSDFComponentType::Any;
};

struct BSDFDualEvalResult
{
    CuDiff::Dual<6, glm::vec3> bsdf_value;
    float sample_probability;
    BSDFComponentType flags = BSDFComponentType::Any;
};

struct BSDFVPtrTable
{
    uint32_t sampleCallIndex;
    uint32_t evalCallIndex;
    uint32_t sampleForwardCallIndex;
    uint32_t evalForwardCallIndex;
    uint32_t sampleBackwardCallIndex;
    uint32_t evalBackwardCallIndex;


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

    __device__ BSDFDualSamplingResult sampleBSDFForward(const DualSurfaceInteraction& si,
                                                        const atcg::SampledWavelengths& wavelengths,
                                                        PCG32& rng) const
    {
        return optixDirectCall<BSDFDualSamplingResult,
                               const DualSurfaceInteraction&,
                               const atcg::SampledWavelengths&,
                               PCG32&>(sampleForwardCallIndex, si, wavelengths, rng);
    }

    __device__ BSDFDualEvalResult evalBSDFForward(const DualSurfaceInteraction& si,
                                                  const CuDiff::Dual<6, glm::vec3>& outgoing_dir,
                                                  const atcg::SampledWavelengths& wavelengths) const
    {
        return optixDirectCall<BSDFDualEvalResult,
                               const DualSurfaceInteraction&,
                               const CuDiff::Dual<6, glm::vec3>&,
                               const atcg::SampledWavelengths&>(evalForwardCallIndex, si, outgoing_dir, wavelengths);
    }

    __device__ void
    evalBSDFBackward(const SurfaceInteraction& si, const glm::vec3& outgoing_dir, const glm::vec3& out_grad) const
    {
        optixDirectCall<void, const SurfaceInteraction&, const glm::vec3&, const glm::vec3&>(evalBackwardCallIndex,
                                                                                             si,
                                                                                             outgoing_dir,
                                                                                             out_grad);
    }

    __device__ void
    sampleBSDFBackward(const SurfaceInteraction& si, PCG32& rng, const glm::vec3& dLdbsdf, const glm::vec3& dLdwo) const
    {
        optixDirectCall<void, const SurfaceInteraction&, PCG32&, const glm::vec3&, const glm::vec3&>(
            sampleBackwardCallIndex,
            si,
            rng,
            dLdbsdf,
            dLdwo);
    }

#endif
};

}    // namespace atcg