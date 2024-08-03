#pragma once

#include <Pathtracing/BSDF.h>

namespace atcg
{
class CPUBSDF : public BSDF
{
public:
    virtual ~CPUBSDF() {}

    virtual BSDFSamplingResult sampleBSDF(const SurfaceInteraction& si, PCG32& rng) const = 0;

    virtual BSDFEvalResult evalBSDF(const SurfaceInteraction& si, const glm::vec3& outgoing_dir) = 0;
};
}    // namespace atcg