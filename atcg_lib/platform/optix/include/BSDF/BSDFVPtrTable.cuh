#pragma once

#include <Core/glm.h>
#include <Core/SurfaceInteraction.h>
#include <Math/Random.h>
#include <BSDF/BSDFFlags.h>
#include <Spectrum/SampledSpectrum.h>
#include <optix.h>
#include <Integrator/TransportMode.h>


namespace atcg
{
struct BSDFSamplingResult
{
    glm::vec3 out_dir;
    SampledSpectrum bsdf_weight;
    float pdf_dw            = 0.0f;
    BSDFComponentType flags = BSDFComponentType::Any;
    float eta               = 1.0f;
};

struct BSDFEvalResult
{
    SampledSpectrum bsdf_value = SampledSpectrum(0);
    float pdf_dw               = 0.0f;
    BSDFComponentType flags    = BSDFComponentType::Any;
};

struct BSDFContext
{
    ATCG_HOST_DEVICE BSDFContext() : transport_mode(TransportMode::IMPORTANCE) {}
    ATCG_HOST_DEVICE BSDFContext(TransportMode mode) : transport_mode(mode) {}

    TransportMode transport_mode = TransportMode::IMPORTANCE;
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
                                             PCG32& rng,
                                             const BSDFContext& ctx = BSDFContext()) const
    {
        return optixDirectCall<BSDFSamplingResult,
                               const SurfaceInteraction&,
                               const atcg::SampledWavelengths&,
                               PCG32&,
                               const BSDFContext&>(sampleCallIndex, si, wavelengths, rng, ctx);
    }

    __device__ BSDFEvalResult evalBSDF(const SurfaceInteraction& si,
                                       const glm::vec3& outgoing_dir,
                                       const atcg::SampledWavelengths& wavelengths,
                                       const BSDFContext& ctx = BSDFContext()) const
    {
        return optixDirectCall<BSDFEvalResult,
                               const SurfaceInteraction&,
                               const glm::vec3&,
                               const atcg::SampledWavelengths&,
                               const BSDFContext&>(evalCallIndex, si, outgoing_dir, wavelengths, ctx);
    }

    __device__ float evalPDF(const SurfaceInteraction& si,
                             const glm::vec3& outgoing_dir,
                             const atcg::SampledWavelengths& wavelengths,
                             const BSDFContext& ctx = BSDFContext()) const
    {
        return optixDirectCall<float,
                               const SurfaceInteraction&,
                               const glm::vec3&,
                               const atcg::SampledWavelengths&,
                               const BSDFContext&>(evalPDFCallIndex, si, outgoing_dir, wavelengths, ctx);
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