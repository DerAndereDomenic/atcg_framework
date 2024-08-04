#pragma once

#include <Pathtracing/BSDF.h>
#include <Pathtracing/Emitter.h>

namespace atcg
{
class CPUBSDF : public BSDF
{
public:
    virtual ~CPUBSDF() {}

    virtual BSDFSamplingResult sampleBSDF(const SurfaceInteraction& si, PCG32& rng) const = 0;

    virtual BSDFEvalResult evalBSDF(const SurfaceInteraction& si, const glm::vec3& outgoing_dir) = 0;
};

class CPUEmitter : public Emitter
{
public:
    virtual ~CPUEmitter() {}

    virtual glm::vec3 evalLight(const SurfaceInteraction& si) const = 0;

    virtual EmitterSamplingResult sampleLight(const SurfaceInteraction& si, PCG32& rng) const = 0;

    virtual PhotonSamplingResult samplePhoton(PCG32& rng) const = 0;

    virtual float evalLightSamplingPdf(const SurfaceInteraction& last_si, const SurfaceInteraction& si) const = 0;

protected:
};
}    // namespace atcg