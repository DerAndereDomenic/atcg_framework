#pragma once

#include <Core/glm.h>
#include <Core/SurfaceInteraction.h>
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
    float pdf_dw            = 0.0f;
    BSDFComponentType flags = BSDFComponentType::Any;
};

struct BSDFEvalResult
{
    SampledSpectrum bsdf_value = SampledSpectrum(0);
    float pdf_dw               = 0.0f;
    BSDFComponentType flags    = BSDFComponentType::Any;
};

struct BSDFVPtrTable
{
    uint32_t sampleCallIndex;
    uint32_t evalCallIndex;
    uint32_t evalPDFCallIndex;

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

    __device__ float evalPDF(const SurfaceInteraction& si,
                             const glm::vec3& outgoing_dir,
                             const atcg::SampledWavelengths& wavelengths) const
    {
        return optixDirectCall<float, const SurfaceInteraction&, const glm::vec3&, const atcg::SampledWavelengths&>(
            evalPDFCallIndex,
            si,
            outgoing_dir,
            wavelengths);
    }

    __device__ inline bool isDelta() const { return hasBSDFFlag(flags, BSDFComponentType::AnyDelta); }
    __device__ inline bool isIdealReflection() const { return hasBSDFFlag(flags, BSDFComponentType::IdealReflection); }
    __device__ inline bool isGlossyReflection() const
    {
        return hasBSDFFlag(flags, BSDFComponentType::GlossyReflection);
    }
    __device__ inline bool isDiffuseReflection() const
    {
        return hasBSDFFlag(flags, BSDFComponentType::DiffuseReflection);
    }
    __device__ inline bool isIdealTransmission() const
    {
        return hasBSDFFlag(flags, BSDFComponentType::IdealTransmission);
    }
    __device__ inline bool isGlossyTransmission() const
    {
        return hasBSDFFlag(flags, BSDFComponentType::GlossyTransmission);
    }
    __device__ inline bool isDiffuseTransmission() const
    {
        return hasBSDFFlag(flags, BSDFComponentType::DiffuseTransmission);
    }
    __device__ inline bool isReflection() const { return hasBSDFFlag(flags, BSDFComponentType::AnyReflection); }
    __device__ inline bool isTransmission() const { return hasBSDFFlag(flags, BSDFComponentType::AnyTransmission); }
    __device__ inline bool isNull() const { return hasBSDFFlag(flags, BSDFComponentType::NullTransmission); }

#endif
};

}    // namespace atcg